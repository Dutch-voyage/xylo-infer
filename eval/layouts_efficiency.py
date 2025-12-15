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

# LS: layerwise selection
# LA: layerwise allocation
# HS: headwise selection
# HA: headwise allocation

torch.set_printoptions(profile="full")

device = "cuda:0"

num_blocks = 2048 * 12 * 36

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
        
        self.kv_last_page_lens = torch.ones(32 * 36, dtype=torch.int32, device="cuda")   

        self.paged_prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            backend="fa2",
        )

        self.ragged_prefill_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, 
            "NHD", 
            backend="auto"
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
            if hasattr(self, "LS_rewrite_global_block_table"):
                self.LS_rewrite_global_block_table[seq_id, :selected_block_table_ids.shape[0]] = selected_block_table_ids.to(torch.int32)
            if hasattr(self, "LS_sparse_global_block_table"):
                self.LS_sparse_global_block_table[seq_id, :block_table_ids.shape[0]] = block_table_ids.to(torch.int32)
            
        cu_mask = torch.cat(masks, dim=0).contiguous().view(-1)
        cu_packed_mask, _ = segment_packbits(cu_mask, mask_indptr, bitorder="little")
        
        self.packed_custom_mask = cu_packed_mask
        self.LS_rewrite_cu_page_lengths = torch.ones(
            (num_seqs, ), dtype=torch.int32, device="cuda"
        ) * num_selected
        
        self.LS_sparse_cu_page_lengths = torch.ones(
            (num_seqs, ), dtype=torch.int32, device="cuda"  
        ) * seq_length
        
    def LS_rewrite_prepare(self):
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
        
    def LS_sparse_prepare(self):
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

        self.paged_prefill_wrapper.plan(
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

        self.forward_wrapper = self.paged_prefill_wrapper
        
    def LA_generate_layouts(self, num_seqs, seq_length, num_selected):
        beta = 36 * (1 / 8) / sum([1 / i for i in range(1, num_layers + 1)])
        nums_selected = [int(seq_length * beta / l) for l in range(1, num_layers + 1)]
        selected_indices_all_layers = torch.cumsum(
            torch.tensor([0] + nums_selected, dtype=torch.int32, device="cuda"), dim=0
        )
        seq_length_all_layers = selected_indices_all_layers[-1]
        masks_sum = []
        mask_indptr_sum = [0] + [seq_length_all_layers] * num_seqs
        mask_indptr_sum = torch.cumsum(torch.tensor(mask_indptr_sum, dtype=torch.int32, device="cuda"), dim=0)
        block_indices_all_layers = torch.arange(0, seq_length_all_layers, device="cuda", dtype=torch.int32)
        
        for seq_id in range(num_seqs):
            block_table_ids = torch.randint(
                0, num_blocks, (seq_length_all_layers,), device="cuda"
            )
            
            selected_indices = block_indices_all_layers[selected_indices_all_layers[self.layer_id]:selected_indices_all_layers[self.layer_id + 1]]
            
            cu_mask_sum = torch.zeros((seq_length_all_layers,), dtype=torch.bool, device="cuda")
            cu_mask_sum[selected_indices] = True
            
            masks_sum.append(cu_mask_sum)   
            
            self.LA_flatten_sum_global_block_table[seq_id, :block_table_ids.shape[0]] = block_table_ids.to(torch.int32)
            
        cu_mask = torch.cat(masks_sum, dim=0).contiguous().view(-1)
        cu_packed_mask_sum, _ = segment_packbits(cu_mask, mask_indptr_sum, bitorder="little")
        
        self.packed_custom_mask = cu_packed_mask_sum
        self.LA_flatten_sum_cu_page_lengths = torch.ones(
            (num_seqs, ), dtype=torch.int32, device="cuda"
        ) * nums_selected[self.layer_id]
        
    def LA_flatten_sum_prepare(self):
        self.cu_page_lengths = self.LA_flatten_sum_cu_page_lengths
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

        self.paged_prefill_wrapper.plan(
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
        
        self.forward_wrapper = self.paged_prefill_wrapper
        
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
    
    def run(self, method, num_blocks, num_seqs_list, seq_length_means):
        stats = []
        if method == "LS_rewrite_prefill":
            # generate_tables = self.generate_block_tables
            self.LS_rewrite_global_block_table = torch.zeros((32, 2048 * 10 * 64), dtype=torch.int32, device="cuda")
            self.global_block_table = self.LS_rewrite_global_block_table
            generate_tables = self.generate_LS_layouts
            prepare = self.LS_rewrite_prepare
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
            self.LS_sparse_global_block_table = torch.zeros((32, 2048 * 10 * 64), dtype=torch.int32, device="cuda")
            self.global_block_table = self.LS_sparse_global_block_table
            generate_tables = self.generate_LS_layouts
            prepare = self.LS_sparse_prepare
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
        elif method == "LA_flatten_sum_prefill":
            self.LA_flatten_sum_global_block_table = torch.zeros((32, 2048 * 10 * 64), dtype=torch.int32, device="cuda")
            self.global_block_table = self.LA_flatten_sum_global_block_table
            generate_tables = self.LA_generate_layouts
            prepare = self.LA_flatten_sum_prepare
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
            self.layer_id = 0
            for _ in range(5):
                with torch.no_grad():
                    generate_tables(num_seqs, 2048, 512)
                    prepare()
                    _ = compute(q, self.k_cache, self.v_cache)

            torch.cuda.synchronize()

            # Timing runs
            prepare_times = []
            compute_times = []
            num_runs = num_layers
            
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
                        self.layer_id += 1

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
    # num_seqs = [1, 2, 4, 8, 16, 32]
    num_seqs = [1]
    # seq_lengths = [i * 2048 for i in range(1, 11)]
    seq_lengths = [2048]
    # seq_lengths = [8]
    LA_flatten_sum_stats = bench.run("LA_flatten_sum_prefill", num_blocks, num_seqs, seq_lengths)
    # LS_rewrite_prefill = bench.run("LS_rewrite_prefill", num_blocks, num_seqs, seq_lengths)
    LS_sparse_prefill = bench.run("LS_sparse_prefill", num_blocks, num_seqs, seq_lengths)
    
    print("LA Flatten Sum Prefill Stats:", LA_flatten_sum_stats)
    # print("LS Rewrite Prefill Stats:", LS_rewrite_prefill)
    print("LS Sparse Prefill Stats:", LS_sparse_prefill)
    
    # np.save("LS_backend_stats.npy", 
    #         {
    #          "LS_rewrite_prefill": LS_rewrite_prefill, 
    #          "LS_sparse_prefill": LS_sparse_prefill
    #         })
    