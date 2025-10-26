from copy import copy
from enum import Enum, auto
from itertools import count

from ..sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 1
    counter = count()
    
    def __init__(self):
        self.block_table: list[int] = []
        self.num_tokens: int = 0
        self.num_prompt_tokens: int = 0
        self.num_cached_tokens: int = 0
    
    @classmethod
    def for_capture(cls, block_table: list[int]):
        seq = cls()
        seq.block_table = block_table
        seq.num_tokens = len(block_table) * cls.block_size
        return seq
    
    @classmethod
    def from_prompt(cls, token_ids: list[int], sampling_params = SamplingParams()):
        seq = cls()
        seq.seq_id = next(Sequence.counter)
        seq.status = SequenceStatus.WAITING
        seq.token_ids = copy(token_ids)
        seq.last_token = token_ids[-1]
        seq.num_tokens = len(seq.token_ids)
        seq.num_prompt_tokens = len(token_ids)
        seq.num_cached_tokens = 0
        seq.block_table = []
        seq.temperature = sampling_params.temperature
        seq.max_tokens = sampling_params.max_tokens
        seq.ignore_eos = sampling_params.ignore_eos
        return seq

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
