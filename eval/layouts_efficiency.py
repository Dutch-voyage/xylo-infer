from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    BatchDecodeWithPagedKVCacheWrapper,
)
from flashinfer.quantization import segment_packbits
import itertools
import torch
import numpy as np
import triton
import triton.language as tl 

torch.set_printoptions(profile="full")

device = "cuda:0"

num_blocks = 2048 * 128

num_kv_heads = 8
num_heads = 32
num_layers = 8
head_dim = 128

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

class BenchMark:
    def __init__(self):
        self.workspace_buffer = torch.empty(
            512 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        )
        
        self.rewrite_global_block_table = torch.zeros((32, 2048 * 10 * 64), dtype=torch.int32, device="cuda")
        self.sparse_global_block_table = torch.zeros((32, 2048 * 10 * 64), dtype=torch.int32, device="cuda")
        
        self.kv_last_page_lens = torch.ones(32 * 36, dtype=torch.int32, device="cuda")   

        self.paged_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            # use_tensor_cores=True,
        )

        self.paged_prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            backend="fa2",
        )

        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, 
            "NHD", 
            backend="auto"
        )
        
        self.sparse_decode_prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            backend="fa2",
        )
        
        self.layerwise_sparse_decode_prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            backend="fa2",
        )
    
    def generate_LS_layouts(self, num_seqs, seq_length, num_selected):
        masks = []
        mask_indptr = [0] + [seq_length] * num_seqs
        mask_indptr = torch.cumsum(torch.tensor(mask_indptr, dtype=torch.int32, device="cuda"), dim=0)
        for seq_id in range(num_seqs):
            block_table_ids = torch.randint(
                0, num_blocks, (seq_length,), device="cuda"
            )
            selected_indices = torch.randint(
                0, seq_length, (num_selected,), device="cuda"
            )
            cu_mask = torch.zeros((seq_length,), dtype=torch.bool, device="cuda")
            cu_mask[selected_indices] = True
            
            masks.append(cu_mask)
            
            selected_block_table_ids = block_table_ids[selected_indices]
            # Write to global block table: [seq_id, seq_length] = block_table_id using tensor indexing
            self.rewrite_global_block_table[seq_id, :selected_block_table_ids.shape[0]] = selected_block_table_ids.to(torch.int32)
            self.sparse_global_block_table[seq_id, :block_table_ids.shape[0]] = block_table_ids.to(torch.int32)
            
        cu_mask = torch.cat(masks, dim=0).contiguous().view(-1)
        cu_packed_mask, _ = segment_packbits(cu_mask, mask_indptr, bitorder="little")
        
        self.packed_custom_mask = cu_packed_mask
        self.LS_rewrite_cu_page_lengths = torch.ones(
            (num_seqs, ), dtype=torch.int32, device="cuda"
        ) * num_selected
        
        self.LS_sparse_cu_page_lengths = torch.ones(
            (num_seqs, ), dtype=torch.int32, device="cuda"  
        ) * seq_length
        
    def LS_rewrite_prefill_prepare(self):
        self.cu_page_lengths = self.LS_rewrite_cu_page_lengths
        num_reqs = self.cu_page_lengths.shape[0]
        kv_indptr = torch.zeros((self.cu_page_lengths.shape[0] + 1), device="cuda", dtype=torch.int32)
        kv_indptr[1:] = torch.cumsum(self.cu_page_lengths, dim=0)

        kv_page_indices = torch.empty(
            kv_indptr[-1], dtype=torch.int32, device="cuda"
        )

        create_flashinfer_kv_indices_triton[(num_reqs,)](
            self.global_block_table,
            torch.arange(num_reqs, device="cuda", dtype=torch.int32),
            self.cu_page_lengths,
            kv_indptr,
            None,
            kv_page_indices,
            self.global_block_table.shape[1],
        )

        self.cu_kv_page_indices  = torch.arange(0, kv_page_indices.shape[0], device="cuda").to(torch.int32)

        self.k_cache[self.cu_kv_page_indices] = self.k_cache[kv_page_indices]# .clone()
        self.v_cache[self.cu_kv_page_indices] = self.v_cache[kv_page_indices]# .clone()
        
        kv_last_page_lens = self.kv_last_page_lens[:num_reqs]   
        qo_indptr = torch.arange(num_reqs + 1, device="cuda").to(torch.int32)

        self.paged_prefill_wrapper.begin_forward(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_page_indices,
            paged_kv_last_page_len=kv_last_page_lens,
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            causal=True,
            page_size=1,
            q_data_type=torch.bfloat16,
        )
        self.forward_wrapper = self.paged_prefill_wrapper
        
    def LS_sparse_prefill_prepare(self):
        self.cu_page_lengths = self.LS_sparse_cu_page_lengths
        num_reqs = self.cu_page_lengths.shape[0]
        # Use the cu_page_lengths already computed in generate_sparse_block_table
        kv_indptr = torch.zeros((self.cu_page_lengths.shape[0] + 1), device="cuda", dtype=torch.int32)
        kv_indptr[1:] = torch.cumsum(self.cu_page_lengths, dim=0)
        
        kv_page_indices = torch.empty(
            kv_indptr[-1], dtype=torch.int32, device="cuda"
        )
        
        create_flashinfer_kv_indices_triton[(num_reqs,)](
            self.global_block_table,
            torch.arange(num_reqs, device="cuda", dtype=torch.int32),
            self.cu_page_lengths,
            kv_indptr,
            None,
            kv_page_indices,
            self.global_block_table.shape[1],
        )        
        
        kv_last_page_lens = self.kv_last_page_lens[:num_reqs]

        qo_indptr = torch.arange(num_reqs + 1, device="cuda").to(torch.int32)

        self.sparse_decode_prefill_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_page_indices,
            paged_kv_last_page_len=kv_last_page_lens,
            num_qo_heads=num_heads // num_kv_heads,
            num_kv_heads=1,
            head_dim_qk=head_dim,
            # causal=True,
            # custom_mask=mask,
            packed_custom_mask=self.packed_custom_mask,
            page_size=1,
            q_data_type=torch.bfloat16,
        )

        self.forward_wrapper = self.sparse_decode_prefill_wrapper
        
    def flatten_compute(self, q, k_cache, v_cache):
        q = q.view(-1, num_kv_heads, num_heads // num_kv_heads, head_dim).reshape(-1, num_heads // num_kv_heads, head_dim)
        o = self.forward_wrapper.forward(q, (k_cache, v_cache))
        o = o.view(-1, num_kv_heads, num_heads // num_kv_heads, head_dim).view(-1, num_heads, head_dim)
        return o 
    
    def ragged_compute(self, q, k_cache, v_cache ):
        key, value = k_cache[:self.cu_kv_page_indices[-1]], v_cache[:self.cu_kv_page_indices[-1]]
        o = self.forward_wrapper.forward(q, key, value, causal=True)
        return o
    
    def compute(self, q, k_cache, v_cache):
        o = self.forward_wrapper.forward(q, (k_cache, v_cache))
        return o
    
    def layerwise_generate_sparse_block_table(self, num_seqs, seq_length_mean):
        block_tables = []
        layerwise_masks = []

        for seq_id in range(num_seqs):
            block_table_ids = torch.randint(
                0, num_blocks, (seq_length_mean,), device="cuda"
            ).tolist()
            layerwise_mask = torch.full((num_layers,), False, device="cuda", dtype=torch.bool)
            layerwise_mask[0] = True
            layerwise_masks.append(layerwise_mask.unsqueeze(0).repeat(len(block_table_ids), 1).view(-1))
            cu_block_table = list(itertools.chain(*[[table_id * num_layers + i for i in range(num_layers)] for table_id in block_table_ids]))
            assert np.max(block_table_ids) * num_layers < num_blocks * num_layers, "Block ID exceeds number of blocks"
            block_tables.append(cu_block_table)

            # Write to global block table: [seq_id, seq_length] = block_table_id using tensor indexing
            block_table_tensor = torch.tensor(cu_block_table, device="cuda")
            self.global_block_table[seq_id, :len(cu_block_table)] = block_table_tensor
        
        mask_indptr = torch.cumsum(
            torch.tensor(
                [0] + [len(block_table) for block_table in block_tables], device="cuda"
            ),
            dim=0,
        ).to(torch.int32)
        
        custom_mask = torch.cat(layerwise_masks, dim=0).contiguous().view(-1)
        # print(custom_mask.shape)
        packed_custom_mask, _ = segment_packbits(custom_mask, mask_indptr, bitorder="little")
        
        self.cu_page_lengths = torch.tensor(
            [len(block_table) for block_table in block_tables], device="cuda"
        ).to(torch.int32)
        
        return (block_tables, packed_custom_mask)

    def run(self, method, num_blocks, num_seqs_list, seq_length_means):
        stats = []
        if method == "LS_rewrite_prefill":
            # generate_tables = self.generate_block_tables
            self.global_block_table = self.rewrite_global_block_table
            generate_tables = self.generate_LS_layouts
            prepare = self.LS_rewrite_prefill_prepare
            self.k_cache = torch.randn(
                [num_blocks, num_kv_heads, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            self.v_cache = torch.randn(
                [num_blocks, num_kv_heads, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            
            compute = self.compute
        elif method == "LS_sparse_prefill":
            self.global_block_table = self.sparse_global_block_table
            generate_tables = self.generate_LS_layouts
            prepare = self.LS_sparse_prefill_prepare
            self.k_cache = torch.randn(
                [num_blocks, num_kv_heads, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            self.v_cache = torch.randn(
                [num_blocks, num_kv_heads, head_dim],
                dtype=torch.bfloat16,
                device=device,
            )
            compute = self.compute
        else:
            raise ValueError(f"Unknown method: {method}")
            
        for num_seqs in num_seqs_list:
            q = torch.randn([num_seqs, 32, 128], dtype=torch.bfloat16).to(device)
            # Warmup runs
            for _ in range(5):
                with torch.no_grad():
                    generate_tables(num_seqs, 2048, 512)
                    prepare()
                    _ = compute(q, self.k_cache, self.v_cache)

            torch.cuda.synchronize()

            # Timing runs
            prepare_times = []
            compute_times = []
            num_runs = 10
            
            print("Starting benchmark...")
            with torch.no_grad():
                for seq_length_mean in seq_length_means:
                    for _ in range(num_runs):
                        generate_tables(num_seqs, seq_length_mean, seq_length_mean // 8)
                        prepare_start_event = torch.cuda.Event(enable_timing=True)
                        prepare_end_event = torch.cuda.Event(enable_timing=True)

                        prepare_start_event.record()
                        prepare()
                        prepare_end_event.record()
                        torch.cuda.synchronize()

                        prepare_times.append(
                            prepare_start_event.elapsed_time(prepare_end_event)
                        )

                        compute_start_event = torch.cuda.Event(enable_timing=True)
                        compute_end_event = torch.cuda.Event(enable_timing=True)

                        compute_start_event.record()
                        output = compute(q, self.k_cache, self.v_cache)
                        compute_end_event.record()
                        torch.cuda.synchronize()

                        compute_times.append(
                            compute_start_event.elapsed_time(compute_end_event)
                        )

                    prepare_mean_ms = np.mean(prepare_times)
                    compute_mean_ms = np.mean(compute_times)
                    stats.append({
                        "seq_length_mean": seq_length_mean,
                        "num_seqs": num_seqs,
                        "prepare_mean_ms": prepare_mean_ms,
                        "compute_mean_ms": compute_mean_ms,
                    })

        return stats


if __name__ == "__main__":
    # set random seed
    np.random.seed(1111)
    bench = BenchMark()
    num_seqs = [1, 2, 4, 8, 16, 32]
    # num_seqs = [32]
    seq_lengths = [i * 2048 for i in range(1, 11)]
    # seq_lengths = [8]
    LS_rewrite_prefill = bench.run("LS_rewrite_prefill", num_blocks, num_seqs, seq_lengths)
    LS_sparse_prefill = bench.run("LS_sparse_prefill", num_blocks, num_seqs, seq_lengths)
    
    print("LS Rewrite Prefill Stats:", LS_rewrite_prefill)
    print("LS Sparse Prefill Stats:", LS_sparse_prefill)
    
    np.save("LS_backend_stats.npy", 
            {
             "LS_rewrite_prefill": LS_rewrite_prefill, 
             "LS_sparse_prefill": LS_sparse_prefill
            })
    
    # np.save("backend_stats.npy", 
    #         {
    #         #  "paged_decode": paged_decode_stats,
    #         #  "paged_prefill": paged_prefill_stats,
    #         #  "sparse_prefill": sparse_prefill_stats,
    #         "ragged_prefill": ragged_prefill_stats, 
    #         "layerwise_sparse_decode_prefill": layerwise_sparse_decode_prefill_stats
    #         })
