import torch
import torch.nn as nn
import torch.nn.functional as F

from src.services.nanovllm_v8.utils.logging import append_item_to_log

from .utils import cal_similarity, compute_attention_scores, update_log, gather_selected_kv
from .lse_preserve_merge import merge_fixed_budget, merge_multi_to_one
from .binary_search import binary_search_T_linear, gradient_descent_T_linear

from flashinfer.sampling import top_p_renorm_probs
from flashinfer.quantization import segment_packbits


class RKV:
    def __init__(
        self,
        config,
        budget=1024,
        lower_bound=128,
        window_size=8,
        kernel_size=7,
        mix_lambda=0.07,
        retain_ratio=0.2,
        retain_direction="last",
        record_kept_token_indices=False,
        **kwargs,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        # self.budget = budget
        self.budget = lower_bound
        self.sink_size = 1
        self.window_size = window_size - self.sink_size
        self.kernel_size = kernel_size
        self.mix_lambda = mix_lambda
        self.retain_ratio = retain_ratio
        self.retain_direction = retain_direction
        
        self.lse_preserve_merge = config.lse_preserve_merge
        self.p_attn = config.p_attn
        self.if_log_compress = config.if_log_compress
        
        self.temperatures = {}
        
    def update_kv(
        self,
        query_states,
        key_states,
        value_states,
        effective_kv_head_lens=None,
        seq_id = None, 
        *args,
    ):
        bsz, num_heads, q_cache_len, head_dim = query_states.shape
        kv_cache_len = key_states.shape[-2]
        num_kv_heads = key_states.shape[1]

        if kv_cache_len < self.budget:
            return {
                "key_states": key_states,
                "value_states": value_states,
            }
        else:
            attn_weights = compute_attention_scores(query_states, key_states)
            if effective_kv_head_lens is not None:
                
                indices = torch.arange(
                    kv_cache_len, device=key_states.device
                ).view(1, 1, -1)
                
                lengths = effective_kv_head_lens.unsqueeze(-1)
                
                effective_mask = indices < lengths.to(indices.device)

                attn_weights = attn_weights.masked_fill(~effective_mask.unsqueeze(2), float("-inf"))

            attn_weights_sum = (
                nn.functional.softmax(
                    attn_weights[:, :, :, self.sink_size : -self.window_size],
                    dim=-1,
                    dtype=torch.float32,
                )
                # .mean(dim=-2)
                .to(query_states.dtype)
            )

            # TODO: Softmax then reduce head

            # attn_cache = F.max_pool1d(
            #     attn_weights_sum,
            #     kernel_size=self.kernel_size,
            #     padding=self.kernel_size // 2,
            #     stride=1,
            # )

            attn_cache = attn_weights_sum

            similarity_cos = -cal_similarity(
                key_states,
                aggregation="none", 
                normalization=True,
                retain_ratio=self.retain_ratio,
                retain_direction=self.retain_direction,
            )[:, self.sink_size : -self.window_size].unsqueeze(0)
            
            shifted_probs = attn_cache * self.mix_lambda + similarity_cos * (
                1 - self.mix_lambda
            )

            # shifted_probs, T = gradient_descent_T_linear(
            #     attn_weights_sum,
            #     shifted_probs,
            #     self.p_attn,
            # )
            
            shifted_probs -= shifted_probs.amin(dim=-1, keepdim=True).detach()
            shifted_probs = shifted_probs / shifted_probs.sum(dim=-1, keepdim=True)
            shift_logits = torch.log(shifted_probs + 1e-10)
    
            if seq_id not in self.temperatures:            
                attn_cache, T = gradient_descent_T_linear(
                    attn_weights_sum,
                    shift_logits,
                    self.p_attn,
                )
                self.temperatures[seq_id] = T
            else:
                T = self.temperatures[seq_id]
                attn_cache = shift_logits / T.unsqueeze(-1)
                attn_cache = F.softmax(attn_cache, dim=-1)

            if self.if_log_compress:
                update_log(
                    attn_cache,
                    key_states,
                    value_states,
                    query_states,
                    self.p_attn,
                    self.sink_size,
                    self.window_size,
                )
                append_item_to_log("temperatures", T.reshape(-1).cpu())
            
            if self.lse_preserve_merge:
                # k_compress, v_compress = merge_fixed_budget(
                #     attn_cache,
                #     attn_weights_sum, 
                #     self.budget - self.window_size - self.sink_size,
                #     key_states[:, :, self.sink_size : -self.window_size, :],
                #     value_states[:, :, self.sink_size : -self.window_size, :],
                # )
                k_compress, v_compress = merge_multi_to_one(
                    attn_cache,
                    attn_weights_sum, 
                    self.budget - self.window_size - self.sink_size,
                    key_states[:, :, self.sink_size : -self.window_size, :],
                    value_states[:, :, self.sink_size : -self.window_size, :],
                )

            else:
                selected_mask_full = torch.zeros(num_kv_heads, kv_cache_len, dtype=torch.bool, device=key_states.device)
                
                attn_topp_normed = top_p_renorm_probs(attn_cache.view(-1, attn_cache.shape[-1]), top_p=self.p_attn)
                
                unselected_mask = (attn_topp_normed == torch.zeros_like(attn_topp_normed))
                
                unselected_mask = unselected_mask.reshape(num_kv_heads, -1)
                
                selected_mask = ~unselected_mask
                
                selected_mask_full[:, self.sink_size : -self.window_size] = selected_mask
                
                k = min(self.budget - self.window_size - self.sink_size, attn_cache.shape[-1])
                # save the top budget indices
                indices_desc_topk = attn_cache.squeeze(0).topk(k, dim=-1).indices
                selected_mask_full.scatter_(-1, indices_desc_topk + self.sink_size, True)
                
                print(selected_mask_full.sum(-1))
                
                selected_mask_full[..., :self.sink_size] = True
                selected_mask_full[..., -self.window_size:] = True
                
                key_states = gather_selected_kv(key_states, selected_mask_full.unsqueeze(0))
                value_states = gather_selected_kv(value_states, selected_mask_full.unsqueeze(0))
                
                num_blocks_head = selected_mask_full.to(torch.int32).sum(-1)
                
                organized_selected_mask = torch.zeros_like(selected_mask_full)
                for head_id in range(num_kv_heads):
                    organized_selected_mask[head_id, :num_blocks_head[head_id]] = 1
                
                mask_indptr = torch.arange(0, num_kv_heads + 1).to(selected_mask.device) * kv_cache_len
                packed_selected_mask, _ = segment_packbits(organized_selected_mask.view(-1), mask_indptr, bitorder="little")
                
                packed_selected_mask = packed_selected_mask.view(8, -1)
                
                key_states = key_states.transpose(1, 2).squeeze(0).contiguous()
                value_states = value_states.transpose(1, 2).squeeze(0).contiguous()
                
                print(key_states.shape)
                    
            return {"key_states": key_states, "value_states": value_states, "packed_selected_mask": packed_selected_mask, "num_blocks_this_layer": num_blocks_head.max().item()}
