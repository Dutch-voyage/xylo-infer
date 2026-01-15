# This version of block manager does not implement with prefix caching
# only support block_size == 1 in this version as well
from collections import deque
import xxhash
import numpy as np

from src.services.nanovllm_v8.engine.sequence import Sequence, torch_rotl_uint8

from src.core.service_base import BaseService

import torch

class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.token_ids = []

    def update(self, token_ids: list[int]):
        self.token_ids = token_ids

    def reset(self):
        self.token_ids = []



class BlockManager(BaseService):
    @property
    def name(self):
        return "BlockManager"

    def __init__(self, num_blocks: int, block_size: int, num_kv_heads: int):
        super().__init__()
        assert num_blocks > 0
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
    
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
        
    def update_blocks_post_compression(self, seq: Sequence):
        for block_id in reversed(seq.block_table[seq.num_blocks:]):
            self._deallocate_block(block_id)
        seq.block_table = seq.block_table[:seq.num_blocks]  
        seq.head_extend_block_table = seq.head_extend_block_table[:seq.num_blocks]
        
        # NOTE need further design
        
        for layer_id in range(Sequence.num_layers):
            seq.headwise_mask_layer_transpose[layer_id] = seq.headwise_mask_layer_transpose[layer_id][:(seq.num_blocks + 7) // 8]

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
            block.update(token_ids)
            seq.block_table.append(block_id)
            seq.head_extend_block_table.append([block_id * self.num_kv_heads + i for i in range(self.num_kv_heads)])
            if seq.next_mask == 0b00000001 and i != 0:
                seq.headwise_mask_layer_transpose = torch.cat(
                    [seq.headwise_mask_layer_transpose, torch.ones((Sequence.num_layers, Sequence.num_kv_heads, 1), device="cpu", dtype=torch.uint8)], dim=2
                )
            else:
                seq.headwise_mask_layer_transpose[:, :, -1] += seq.next_mask
            # for layer_id in range(Sequence.num_layers):
            #     list_ref = seq.headwise_mask_layer_transpose[layer_id]
            #     if seq.next_mask == 0b00000001:
            #         for list_i in list_ref:
            #             list_i.append(seq.next_mask)
            #     else:
            #         for list_i in list_ref:   
            #             list_i[-1] = list_i[-1] + seq.next_mask

            seq.next_mask = torch_rotl_uint8(seq.next_mask, 1)
            # print(seq.next_mask)
            
    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            self._deallocate_block(block_id)
        seq.block_table.clear()
        seq.head_extend_block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        # print([self.blocks[index].hash for index in block_table]) 
        # NOTE when the block == 1, the handling logic is different 
        assert self.block_size == 1
        block_id = self.free_block_ids[0]
        seq.num_blocks += 1
        self._allocate_block(block_id)
        seq.block_table.append(block_id)
        seq.head_extend_block_table.append([block_id * self.num_kv_heads + i for i in range(self.num_kv_heads)])
        # there must be at least one token after prefilling (allocate)
        if seq.next_mask == 0b00000001:
            seq.headwise_mask_layer_transpose = torch.cat(
                [seq.headwise_mask_layer_transpose, torch.ones((Sequence.num_layers, Sequence.num_kv_heads, 1), device="cpu", dtype=torch.uint8)], dim=2
            )
        else:
            seq.headwise_mask_layer_transpose[:, :, -1] += seq.next_mask
        seq.next_mask = torch_rotl_uint8(seq.next_mask, 1)