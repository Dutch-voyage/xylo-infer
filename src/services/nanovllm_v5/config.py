import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    log_path: str = "./logs"
    max_num_batched_tokens: int = 262144
    max_num_seqs: int = 128
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.7
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 1
    query_window_size: int = 64
    layer_budget: int = 320
    num_kvcache_blocks: int = -1

    steps_between_cache_compressions: int = 1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        # assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
