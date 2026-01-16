from src.core.artifact_base import Artifact
from src.core.service_base import BaseService

from ..attention.flashinfer_attention_headflatten import (
    Attention,
    store_kvcache,
    read_kvcache,
    read_q_cache,
)
from src.services.nanovllm_v8.engine.sequence import Sequence
import torch

import itertools

from src.services.nanovllm_v8.utils.context import get_context, set_context_replace
from src.services.nanovllm_v8.utils.logging import get_log, set_log
# all implemntation here

import triton
import triton.language as tl

# Run optimized triton kernel
def grid(batch_size, extend_len, BLOCK_SIZE):
    num_token_blocks = triton.cdiv(extend_len, BLOCK_SIZE)
    return (batch_size, num_token_blocks)

@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        # index into req_to_token_ptr needs to be int64
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)


@triton.jit
def write_req_to_token_pool_triton_optimize_headwise(
    req_to_token_ptr,  # [max_batch * num_kv_heads, max_context_len * num_kv_heads] # allocated by head
    req_pool_indices,
    # pre_lens,
    seq_lens,
    # extend_lens,
    out_cache_loc, # allocated by token 
    num_kv_heads: tl.constexpr,
    req_to_token_ptr_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_token = tl.program_id(1)

    req_pool_index = tl.load(req_pool_indices + pid_batch)
    # pre_len = tl.load(pre_lens + pid_batch)
    seq_len = tl.load(seq_lens + pid_batch)
    # extend_len = seq_len - pre_len
    extend_len = seq_len
    
    cumsum_start = 0
    for i in range(pid_batch):
        # cumsum_start += tl.load(extend_lens + i)
        cumsum_start += tl.load(seq_lens + i)
    
    token_start = pid_token * BLOCK_SIZE

    offset = tl.arange(0, BLOCK_SIZE)
    actual_offset = token_start + offset
    mask = actual_offset < extend_len
    
    # offset_store = offset[None, :] # + tl.arange(0, num_kv_heads)[:, None]
    # actual_offset_store = token_start + offset_store
    # mask_store = actual_offset_store < extend_len
    
    # actual_offset_store should be (num_kv_heads, BLOCK_SIZE)
    
    # actual_offset_store = tl.reshape(actual_offset_store, (BLOCK_SIZE), can_reorder=True)
    
    # mask_store = tl.broadcast_to(tl.reshape(mask_store, (BLOCK_SIZE * num_kv_heads), can_reorder=True)[None, :], (num_kv_heads, BLOCK_SIZE * num_kv_heads))
    
    src_ptr = out_cache_loc + cumsum_start + actual_offset
    src_ptr = tl.max_contiguous(tl.multiple_of(src_ptr, BLOCK_SIZE), BLOCK_SIZE)
    value = tl.load(src_ptr, mask=mask)
    
    value_store = value[None, :] * num_kv_heads + tl.arange(0, num_kv_heads)[:, None]
    
    dst_ptr = (
        req_to_token_ptr
        + req_pool_index * req_to_token_ptr_stride * num_kv_heads
        # + actual_offset_store
        + actual_offset
        # + pre_len
    )
    # dst_ptr = tl.max_contiguous(tl.multiple_of(dst_ptr, BLOCK_SIZE), num_kv_heads * BLOCK_SIZE)

    dst_ptr = dst_ptr[None, :] + tl.arange(0, num_kv_heads)[:, None] * req_to_token_ptr_stride

    tl.store(dst_ptr, value_store, mask=mask[None, :])

def write_req_to_token_pool_headwise(
    req_to_token,  # [max_batch, max_context_len]
    req_pool_indices,
    # pre_lens,
    seq_lens,
    # extend_lens,
    out_cache_loc,
    num_kv_heads, 
    batch_size,
    req_pool_stride,
):
    # Run optimized triton kernel
    # max_extend_len = extend_lens.max().item()
    # NOTE: simplify
    max_extend_len = seq_lens.max().item()
    write_req_to_token_pool_triton_optimize_headwise[grid(batch_size, max_extend_len, 512 // num_kv_heads)](
        req_to_token,
        req_pool_indices,
        # pre_lens,
        seq_lens,
        # extend_lens,
        out_cache_loc,
        num_kv_heads, 
        req_pool_stride,
        BLOCK_SIZE=512 // num_kv_heads, 
    )

class CacheManager(BaseService):
    @property
    def name(self):
        return "CacheManagerHeadwise"

    """
    This version of implementation only 
    """

    def __init__(self, attention_backend: Artifact, config, compressor=None):
        super().__init__()

        attention_backend.register(self)
        
        self.num_kv_heads = config.hf_config.num_key_value_heads 
        
        self.head_dim = config.hf_config.head_dim
        
        self.num_layers = config.hf_config.num_hidden_layers
        
        self.max_context_len = config.max_model_len 
        
        self.max_num_seqs = config.max_num_seqs
        
        self.layer_budget = config.layer_budget

        self.cu_seq_pool_id = 0 
        
        self.seq_to_pool_id = {} 

        self.seq_to_slot_pool = torch.zeros((config.max_num_seqs * self.num_kv_heads, config.max_model_len * self.num_kv_heads), dtype=torch.int32, device="cuda")
        
        self.head_indices = torch.arange(0, self.num_kv_heads, dtype=torch.int32, device="cuda")
        
        self.cu_page_indices = self.cu_seq_lens = None
        
        self.full_headwise_mask_per_token = torch.tensor([2 ** i for i in range(self.num_kv_heads)], device="cuda", dtype=torch.uint8)

        self.compressor = compressor
        
        self.if_fake_compress = config.if_fake_compress
        
        self.mask_arr = {layer_id: [] for layer_id in range(self.num_layers)}
        self.mask_indptr = {layer_id: None for layer_id in range(self.num_layers)}
        self.cu_packed_custom_mask = {layer_id: None for layer_id in range(self.num_layers)}
        
    def allocate_prefill_page_indices(self, seqs: list[Sequence]):
        self.cu_seqs = seqs
        for seq_id in seqs:
            self.seq_to_pool_id[seq_id.seq_id] = self.cu_seq_pool_id
            self.cu_seq_pool_id += 1
            self.cu_seq_pool_id %= self.max_num_seqs
        
        self.cu_seqs_to_slot_pool_indices = torch.tensor([self.seq_to_pool_id[seq_id.seq_id] for seq_id in seqs], device="cuda", dtype=torch.int32)
        
        self.len_seqs = len(seqs)
        
        self.seq_lens = torch.tensor(
            [seq.num_blocks for seq in self.cu_seqs], #  * self.num_kv_heads, 
            device="cuda"
        ).to(torch.int32)
        
        context = get_context()
        
        cu_slot_mapping = context.slot_mapping

        write_req_to_token_pool_headwise(
            self.seq_to_slot_pool,
            self.cu_seqs_to_slot_pool_indices,
            self.seq_lens, 
            cu_slot_mapping, # 
            self.num_kv_heads, 
            self.len_seqs,
            self.max_context_len * self.num_kv_heads,
        ) 
            
    def update_masks_optimized(self, seqs):
        # for layer_id in range(self.num_layers):

        #     # this is the bottleneck
        #     self.mask_arr[layer_id] = [torch.tensor(seq.headwise_mask_layer_transpose[layer_id], device="cuda", dtype=torch.uint8) for seq in seqs]
                                                
        #     self.cu_packed_custom_mask[layer_id] = torch.cat([m.contiguous().view(-1) for m in self.mask_arr[layer_id]], dim=0)

        self.cu_packed_custom_mask_optimized = torch.cat([seq.headwise_mask_layer_transpose.view(self.num_layers, -1).to(device="cuda", dtype=torch.uint8) for seq in seqs], dim=-1)
        
        context = get_context()
        context.packed_headwise_mask = self.cu_packed_custom_mask_optimized
        set_context_replace(context)
        
    def allocate_decode_page_indices(self, seqs: list[Sequence]):
        # sglang says when using overlap mode, should not in-place operation, need to investigate, here is non-overlap mode 
        self.cu_seqs = seqs
        
        self.cu_seqs_to_slot_pool_indices = torch.tensor([self.seq_to_pool_id[seq_id.seq_id] for seq_id in seqs], device="cuda", dtype=torch.int32)
        
        self.len_seqs = len(seqs)
        
        self.seq_lens = torch.tensor(
            [seq.num_blocks for seq in self.cu_seqs], #  * self.num_kv_heads, 
            device="cuda"
        ).to(torch.int32)

        context = get_context()
        
        cu_slot_mapping = context.slot_mapping
        
        cu_slot_mapping_headwise = ((cu_slot_mapping * self.num_kv_heads)[:, None] + self.head_indices[None, :])
        
        seq_to_pool_indices_headwise = (self.cu_seqs_to_slot_pool_indices[:, None] * self.num_kv_heads) + self.head_indices[None, :]
        
        seq_lens_headwise = ((self.seq_lens - 1)[:, None]) # + head_indices[None, :]
        
        self.seq_to_slot_pool[seq_to_pool_indices_headwise, seq_lens_headwise] = cu_slot_mapping_headwise
        
        self.cu_qo_indptr = torch.arange(self.len_seqs * self.num_kv_heads + 1, device="cuda", dtype=torch.int32) 
        
        self.cu_kv_page_lengths = (self.seq_lens).repeat_interleave(self.num_kv_heads)
        
        self.cu_kv_indptr = torch.zeros(self.len_seqs * self.num_kv_heads + 1, device="cuda", dtype=torch.int32)
        
        self.cu_kv_indptr[1:] = torch.cumsum(self.cu_kv_page_lengths, dim=0)
        
        self.cu_kv_page_indices = torch.empty(
            self.cu_kv_indptr[-1], dtype=torch.int32, device="cuda"
        )
        
        create_flashinfer_kv_indices_triton[(self.len_seqs * self.num_kv_heads,)](
            self.seq_to_slot_pool,
            torch.arange(0, self.len_seqs * self.num_kv_heads, device="cuda", dtype=torch.int32), 
            self.cu_kv_page_lengths, 
            self.cu_kv_indptr,
            None,
            self.cu_kv_page_indices,
            self.seq_to_slot_pool.shape[1],
        )
        
        print(self.cu_kv_page_indices)
        print("-" * 100)
        
        # self.update_masks(seqs)
        self.update_masks_optimized(seqs)

    def allocate_page_indices_cudagraph(self, seqs: list[Sequence]):
        context = get_context()
        if context.is_prefill:
            self.allocate_prefill_page_indices(seqs)
        else:
            self.allocate_decode_page_indices(seqs)
            
        self.log_occupied_pages(self.cu_kv_page_indices.shape[0])
    
    def allocate_page_indices(self, seqs: list[Sequence]):
        # move to model runner before capturing cuda graph
        self.cu_seqs = seqs
        
        self.seq_lens = torch.tensor(
            [seq.num_blocks for seq in self.cu_seqs]
        ).to(torch.int32)
        
        headwise_seq_table = list(itertools.chain(*[seq.get_headwise_block_table() for seq in seqs]))
        
        self.cu_kv_indptr = torch.cumsum(
            torch.tensor([0] + [len(table) for table in headwise_seq_table], device="cuda"),
            dim=0, dtype=torch.int32
        )
        
        self.cu_kv_page_indices = torch.tensor(
            list(itertools.chain(*headwise_seq_table)), device="cuda", dtype=torch.int32
        )
        
        self.cu_kv_last_page_lens = torch.tensor(
            [1] * (len(seqs) * self.num_kv_heads), device="cuda", dtype=torch.int32
        )

        self.cu_qo_indptr = torch.arange(len(seqs) * self.num_kv_heads + 1, device="cuda", dtype=torch.int32) # .to(torch.int32)
        
        self.cu_packed_custom_mask_optimized = torch.cat([seq.headwise_mask_layer_transpose.view(self.num_layers, -1).to(device="cuda", dtype=torch.uint8) for seq in seqs], dim=-1)

        context = get_context()
        context.packed_headwise_mask = self.cu_packed_custom_mask_optimized
        set_context_replace(context)
        
        self.log_occupied_pages(self.cu_kv_page_indices.shape[0])

    def log_occupied_pages(self, occupied_pages):
        log = get_log()
        log.occupied_pages = occupied_pages
        set_log(log)
        
    def update_indices(self):        
        if get_context().is_prefill:
            self.prepare_metadata_for_attn_prefill(
                self.cu_seqs
            )
        else:
            self.prepare_metadata_for_attn_decode(
                self.cu_seqs
            )

    def update_indices_capture(self, bs: int):
        self.init_forward_metadata_capture_cuda_graph(
            bs,
            self.cu_qo_indptr, 
            self.cu_kv_indptr, 
            self.cu_kv_page_indices,
            self.cu_packed_custom_mask_optimized[0]
        )

    def update_indices_replay(self, bs: int):
        self.init_forward_metadata_replay_cuda_graph(
            bs,
            self.cu_qo_indptr, 
            self.cu_kv_indptr, 
            self.cu_kv_page_indices,
            self.cu_packed_custom_mask_optimized[0]
        )
    
    # def read_and_store_cache_iterative(self, q_cache, k_cache, v_cache, layer_id):
    #     total_steps = (self.cu_seqs[0].num_blocks - self.compressor.budget) // self.config.steps_between_cache_compressions + 1
    #     for i in range(total_steps):
    #         start = i * self.config.steps_between_cache_compressions + self.compressor.budget
    #         end = min(start + self.config.steps_between_cache_compressions, self.cu_seqs[0].num_blocks)
    #         slot_mappings = self.cu_seqs[0].block_table[:self.compressor.budget] + self.cu_seqs[0].block_table[start:end]
    #         assert len(self.cu_seqs) == 1, "Currently only support single request"

    #         slot_mappings_tensor = torch.tensor(slot_mappings, device="cuda").to(
    #             torch.int32
    #         )
            
    #         query_slot_mapping = [self.cu_seqs[0].query_block_id]

    #         query_slot_mapping_tensor = torch.tensor(query_slot_mapping, device="cuda").to(
    #             torch.int32
    #         )

    #         query = read_q_cache(
    #             q_cache=q_cache,
    #             query_slot_mapping=query_slot_mapping_tensor,
    #         )

    #         key, value = read_kvcache(
    #             k_cache=k_cache,
    #             v_cache=v_cache,
    #             slot_mapping=slot_mappings_tensor,
    #             num_kv_heads=self.num_kv_heads, 
    #             head_dim=self.head_dim
    #         )
            
    #         key = key.unsqueeze(0)
    #         value = value.unsqueeze(0)

    #         updated_k, updated_v = self.compressor.update_kv(
    #             query.transpose(1, 2),
    #             key.transpose(1, 2),
    #             value.transpose(1, 2),
    #             self.cu_seqs[0],
    #             layer_id, 
    #         )

    #         key = updated_k.transpose(1, 2).squeeze(0).contiguous()
    #         value = updated_v.transpose(1, 2).squeeze(0).contiguous()

    #         slot_mappings_tensor = slot_mappings_tensor[: key.shape[0]]
            
    #         store_kvcache(
    #             key=key,
    #             value=value,
    #             k_cache=k_cache,
    #             v_cache=v_cache,
    #             slot_mapping=slot_mappings_tensor,
    #         )
    
    def _packed_read_and_store_cache(self, q_cache, k_cache, v_cache, layer_id):        
        # print(f"in compress {id(self.cu_seqs[0])}")
        # print(f"in compress len {self.cu_seqs[0].block_table}")
        # slot_mappings_packed = list(
        #     itertools.chain(
        #         *[seq.block_table for seq in self.cu_seqs]
        #     )
        # )
                
        # slot_mappings_packed_store = list(
        #     itertools.chain(
        #         *[seq.block_table[:self.layer_budget] for seq in self.cu_seqs]
        #     )
        # )
        
        # slot_mappings_packed = torch.tensor(slot_mappings_packed, device="cuda").to(
        #     torch.int32
        # )
        
        # slot_mappings_packed_store = torch.tensor(slot_mappings_packed_store, device="cuda").to(
        #     torch.int32
        # )
        
        slot_mappings_packed = torch.cat([self.seq_to_slot_pool[self.seq_to_pool_id[seq.seq_id], :seq.num_blocks] for seq in self.cu_seqs], dim=0).to(torch.int32)
        
        slot_mappings_packed_store = torch.cat([self.seq_to_slot_pool[self.seq_to_pool_id[seq.seq_id], :self.layer_budget] for seq in self.cu_seqs], dim=0).to(torch.int32)
        
        seq_lens = torch.tensor([0] + [seq.num_blocks for seq in self.cu_seqs], device="cuda").to(
            torch.int32
        )
        seq_lens_cumsum = torch.cumsum(seq_lens, dim=0)
        
        query_slot_mapping = [seq.query_block_id for seq in self.cu_seqs]
        query_slot_mapping_tensor = torch.tensor(query_slot_mapping, device="cuda").to(
            torch.int32
        )
        
        query = read_q_cache(
            q_cache=q_cache,
            query_slot_mapping=query_slot_mapping_tensor,
        )
        
        key, value = read_kvcache(
            k_cache=k_cache,
            v_cache=v_cache,
            slot_mapping=slot_mappings_packed,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim
        )
                        
        updated_k_list = []
        updated_v_list = []
        
        for i, seq in enumerate(self.cu_seqs):
            query_i = query[i: i + 1, :, :, :]
            key_i = key[seq_lens_cumsum[i]: seq_lens_cumsum[i + 1], :, :].unsqueeze(0)
            value_i = value[ seq_lens_cumsum[i]: seq_lens_cumsum[i + 1], :, :].unsqueeze(0)
            
            ret = self.compressor.update_kv(
                query_i.transpose(1, 2),
                key_i.transpose(1, 2),
                value_i.transpose(1, 2),
                layer_id,
            )
            
            updated_k = ret["key_states"]
            updated_v = ret["value_states"]
            
            if "packed_selected_mask" in ret:
                seq.headwise_mask_layer_transpose[layer_id] = (seq.headwise_mask_layer_transpose[layer_id].clone().to("cuda") & ret["packed_selected_mask"])

            updated_k_list.append(updated_k.transpose(1, 2).squeeze(0).contiguous())
            updated_v_list.append(updated_v.transpose(1, 2).squeeze(0).contiguous())
        
        if self.if_fake_compress:
            return 
        
        key = torch.cat(updated_k_list, dim=0)
        value = torch.cat(updated_v_list, dim=0)
        
        store_kvcache(
            key=key,
            value=value,
            k_cache=k_cache,
            v_cache=v_cache,
            slot_mapping=slot_mappings_packed_store,
        )
    
    def read_and_store_cache(self, q_cache, k_cache, v_cache, layer_id):
        return self._packed_read_and_store_cache(q_cache, k_cache, v_cache, layer_id)
