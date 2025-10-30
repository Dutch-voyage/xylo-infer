import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper, BatchPrefillWithRaggedKVCacheWrapper
from flashinfer.decode import _get_range_buf, get_seq_lens, fast_decode_plan

from flashinfer.cascade import merge_state


import itertools
from typing import Optional, Union


from src.services.nanovllm_v2_5.utils.context import get_context
from src.services.nanovllm_v2_5.engine.sequence import Sequence

from src.core.artifact_base import Artifact
from src.core.service_base import BaseService

from functools import partial 
from dataclasses import dataclass


@dataclass
class PrefillMetadata:
    use_ragged: bool
    no_prefix: bool

global_workspace_buffer = None

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module, Artifact):

    @property
    def name(self):
        return "VanillaAttention"
    
    def __init__(
        self,
        model_runner, 
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        Artifact.__init__(self)
        print("initializing attention")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads

        global global_workspace_buffer
        if global_workspace_buffer is None:
            global_workspace_buffer = torch.empty(
                512 * 1024 * 1024, dtype=torch.uint8, device="cuda"
            )
        self.workspace_buffer = global_workspace_buffer
        
        max_bs = min(model_runner.config.max_num_seqs, 512)
        
        self.qo_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=model_runner.device
        )
        
        self.kv_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=model_runner.device
        )
        
        self.kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )
        
        self.cuda_graph_kv_indices = torch.zeros(
            model_runner.config.hf_config.max_position_embeddings * max_bs, 
            dtype=torch.int32,
            device=model_runner.device
        ) 
        
        self.prefill_wrapper_paged = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, 
            "NHD", 
            backend="fa2"
        )
        
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, 
            "NHD", 
            backend="auto"
        )
        
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            use_tensor_cores=True, 
        )
        
        self.decode_cuda_graph_metadata = {}
               
    def register_for_attn(self, service: BaseService):
        methods_to_register = ["attn"]
        for method in methods_to_register:
            self._register_method(method, service)
    
    def register_for_runner(self, service: BaseService):
        methods_to_regsiter = ["prepare_metadata_for_attn_decode", "prepare_metadata_for_attn_prefill"]
        for method in methods_to_regsiter:
            self._register_method(method, service)
    
    def prepare_metadata_for_attn_prefill(self, seqs: list[Sequence]):
        context = get_context()
        cu_seqlens_q = context.cu_seqlens_q
        
        self.qo_indptr[0: len(seqs) + 1] = cu_seqlens_q
        qo_indptr = self.qo_indptr[0: len(seqs) + 1]
    
        
        # kv_indptr = torch.cumsum(
        #     torch.tensor([0] + [len(seq.block_table) for seq in seqs], device="cuda"),
        #     dim=0,
        # ).to(torch.int32)
        # kv_page_indices = torch.tensor(
        #     list(itertools.chain(*[seq.block_table for seq in seqs])), device="cuda"
        # ).to(torch.int32)
        # kv_last_page_lens = torch.tensor(
        #     [seq.last_block_num_tokens for seq in seqs], device="cuda"
        # ).to(torch.int32)
        
        kv_indptr = torch.cumsum(
            torch.tensor([0] + [seq.num_cached_blocks for seq in seqs], device="cuda"),
            dim=0,
        ).to(torch.int32)
        kv_page_indices = torch.tensor(
            list(itertools.chain(*[seq.block_table[:seq.num_cached_blocks] for seq in seqs])), device="cuda"
        ).to(torch.int32)
        kv_last_page_lens = torch.tensor(
            [seq.last_block_num_tokens for seq in seqs], device="cuda"
        ).to(torch.int32)
        
        self.prefill_metadata = PrefillMetadata(use_ragged=True, no_prefix=context.no_prefix)
        self.prefill_wrapper_ragged.begin_forward(
            qo_indptr=qo_indptr,
            kv_indptr=qo_indptr, 
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim, 
            causal=True, 
            q_data_type=torch.bfloat16,
        )
        
        self.prefill_wrapper_paged.begin_forward(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_page_indices,
            paged_kv_last_page_len=kv_last_page_lens,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            causal=False, 
            page_size=self.block_size,
            q_data_type=torch.bfloat16,
        )
    
    def prepare_metadata_for_attn_decode(self, seqs: list[Sequence]):
        """See https://docs.flashinfer.ai/tutorials/kv_layout.html#page-table-layout for metadata required for flashinfer kernel"""
        kv_indptr = torch.cumsum(
            torch.tensor([0] + [len(seq.block_table) for seq in seqs], device="cuda"),
            dim=0,
        ).to(torch.int32)
        kv_page_indices = torch.tensor(
            list(itertools.chain(*[seq.block_table for seq in seqs])), device="cuda"
        ).to(torch.int32)
        kv_last_page_lens = torch.tensor(
            [seq.last_block_num_tokens for seq in seqs], device="cuda"
        ).to(torch.int32)
    
        self.decode_wrapper.plan(
            indptr=kv_indptr,
            indices=kv_page_indices,
            last_page_len=kv_last_page_lens,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=self.block_size,
            pos_encoding_mode="NONE",
            q_data_type=torch.bfloat16,
        )
        self.forward_wrapper = self.decode_wrapper
    
    def update_indices(self, 
                       bs: int, 
                       decode_wrapper: BatchDecodeWithPagedKVCacheWrapper, 
                       cu_page_indices: torch.Tensor, 
                       seq_lens: torch.Tensor, 
                       ):
        self.kv_indptr[: bs + 1] = torch.cumsum(seq_lens, dim=0)
        kv_indptr = self.kv_indptr[: bs + 1]
        
        kv_indices = decode_wrapper._paged_kv_indices_buf
        kv_indices[: cu_page_indices.shape[0]] = cu_page_indices
        
        decode_wrapper.begin_forward(
            indptr=kv_indptr,
            indices=kv_indices,
            last_page_len=self.kv_last_page_len[:bs],
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=self.block_size,
            q_data_type=torch.bfloat16, 
            non_blocking=True,
        )
        
        
    def init_forward_metadata_capture_cuda_graph(
        self, 
        bs: int, 
        seq_lens: torch.Tensor, 
        cu_page_indices: torch.Tensor, 
    ):
        decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            use_cuda_graph=True, 
            use_tensor_cores=True, 
            paged_kv_indptr_buffer=self.kv_indptr[:bs + 1],
            paged_kv_indices_buffer=self.cuda_graph_kv_indices, 
            paged_kv_last_page_len_buffer=self.kv_last_page_len[:bs] 
        )
        self.update_indices(
            bs, 
            decode_wrapper, 
            cu_page_indices, 
            seq_lens
        )
        # TODO look into sglang's patch to find why there is an performance gain in flashinfer plan
        # decode_wrapper.begin_forward = partial(
        #     fast_decode_plan, decode_wrapper
        # )
        self.decode_cuda_graph_metadata[bs] = decode_wrapper
        self.forward_wrapper = decode_wrapper
    
    def init_forward_metadata_replay_cuda_graph(
        self, 
        bs: int, 
        seq_lens: torch.Tensor,  
        cu_page_indices: torch.Tensor, 
    ):
        self.update_indices(
            bs, 
            self.decode_cuda_graph_metadata[bs], 
            cu_page_indices, 
            seq_lens[:bs + 1]
        )

    def attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor): 
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        
         
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            # o = self.prefill_wrapper_paged.forward(
            #     q, (self.k_cache, self.v_cache), causal=True, sm_scale=self.scale
            # )
            
            if self.prefill_metadata.no_prefix:
                o = self.prefill_wrapper_ragged.forward(
                    q=q, 
                    k=k, 
                    v=v, 
                    causal=True,
                    sm_scale=self.scale, 
                )
            else:
                o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
                    q, k, v, causal=True, sm_scale=self.scale,
                )
                o2, s2 = self.prefill_wrapper_paged.forward_return_lse(
                    q, (self.k_cache, self.v_cache), causal=False, sm_scale=self.scale,
                )
                
                o, _ = merge_state(o1, s1, o2, s2)
        else:    # decode
            # self.prepare_metadata(seqs)
            o = self.forward_wrapper.forward(q, (self.k_cache, self.v_cache))
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
