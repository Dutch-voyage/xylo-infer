from dataclasses import dataclass
import torch

CUDA_GRAPH_ENABLED = False

def set_cuda_graph_flag():
    global CUDA_GRAPH_ENABLED
    CUDA_GRAPH_ENABLED = True
    
def get_cuda_graph_flag():
    global CUDA_GRAPH_ENABLED
    return CUDA_GRAPH_ENABLED

@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    query_slot_mapping: torch.Tensor | None = None
    query_window_pos: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, query_slot_mapping=None, query_window_pos=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, query_slot_mapping, query_window_pos)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
