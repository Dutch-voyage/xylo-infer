import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, einsum

import math

class OrthMerging:
    def __init__(
        self,
        budget=128,
        window_size=8,
    ):
        self.budget = budget
        self.window_size = window_size  
        self.seq_to_per_layer_buffer = {}
        
    def update_kv(
        self, 
        query_states, 
        key_states, 
        value_states,
        # seq_id,
        # layer_id,  
    ): 
        # if seq_id not in self.seq_to_per_layer_buffer:
        #     self.seq_to_per_layer_buffer[seq_id] = {
        #         layer_id: []
        #         for layer_id in range(self.num_layers)
        #     }
        
        bsz, num_q_heads, q_len, head_dim = query_states.shape
        
        _, num_kv_heads, kv_len, _ = key_states.shape

        if kv_len < self.budget:
            return key_states, value_states
        
        topk = kv_len - self.budget
        
        assert topk * 2 <= kv_len, "budget too small compared to current kv length"

        query_states = rearrange(
            query_states,
            "b (h g) l d -> b g h l d", g = num_q_heads // num_kv_heads
        )
        
        # apply mean pooling over query groups
        score = torch.einsum(
            "b g h l d, b h m d -> b g h l m",
            query_states,
            key_states,
        ).mean(dim=1) / math.sqrt(head_dim)
        
        score = torch.exp(score - score.amax(dim=-1, keepdim=True))
        
        # option 1: compute kv_len * kv_len matrix, here orthogonal means the the mininum index in this matrix, 
        
        # option 2: select the maximum and minimum score after pooling on q_len .
        score = score.mean(dim=-2)  # shape: (bsz, num_kv_heads, kv_len)        
        
        max_indices = score.topk(topk, dim=-1).indices  # shape: (bsz, num_kv_heads, topk)
        min_indices = (-score).topk(topk, dim=-1).indices  #
        
        selected_indices = torch.cat([max_indices, min_indices], dim=-1)  # shape: (bsz, num_kv_heads, topk * 2)

        unselected_mask = torch.ones((bsz, num_kv_heads, kv_len), device=key_states.device, dtype=torch.bool)
        unselected_mask.scatter_(-1, selected_indices, 0)
        unselected_indices = unselected_mask.nonzero(as_tuple=True)
                
        k_kept = key_states[unselected_indices].view(bsz, num_kv_heads, self.budget - topk, head_dim)
        v_kept = value_states[unselected_indices].view(bsz, num_kv_heads, self.budget - topk, head_dim)

        score_max = torch.gather(score, -1, max_indices).unsqueeze(-1)
        score_min = torch.gather(score, -1, min_indices).unsqueeze(-1)


        k_merged = (score_max / score_max + score_min) * torch.gather(key_states, 2, max_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)) \
            + (score_min / score_max + score_min) * torch.gather(key_states, 2, min_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))

        # score_merged = torch.einsum(
        #     "b g h l d, b h m d -> b g h l m",
        #     query_states, 
        #     k_merged, 
        # ).mean(dim=1) / math.sqrt(head_dim)
        
        # score_merged = torch.exp(score_merged - score_merged.amax(dim=2, keepdim=True)).mean(dim=-2).unsqueeze(-1)
        
        v_merged = (score_max / score_max + score_min) * torch.gather(value_states, 2, max_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)) \
            + (score_min / score_max + score_min) * torch.gather(value_states, 2, min_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))

        k_final = torch.cat([k_kept, k_merged], dim=2)
        v_final = torch.cat([v_kept, v_merged], dim=2)
                
        return k_final, v_final