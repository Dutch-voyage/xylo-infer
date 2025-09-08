import torch

from src.artifacts.swift.model_runner.utils import LlamaModelConfig, LlamaWeight
from src.artifacts.swift.model_runner.kernels.rmsnorm import rmsnorm_inplace
from src.artifacts.swift.model_runner.kernels.linear import linear
from src.artifacts.swift.model_runner.utils import LlamaInferState

class LlamaPostLayer:
    def __init__(
        self,
        model_config: LlamaModelConfig,
        weights: LlamaWeight,
    ):
        self.model_config = model_config
        self.weights = weights
    
    def forward(
        self,
        input_embds: torch.Tensor,	# [num_total_tokens, hidden_size]
        infer_state: LlamaInferState
    ) -> torch.Tensor:
        # Slice to get the last token embedding for each request
        last_token_indices = torch.cat(
            (
                infer_state.prefill_seq_start_locs + infer_state.prefill_seq_lens - 1,
                torch.arange(infer_state.num_prefill_tokens, infer_state.num_tokens, device=input_embds.device, dtype=torch.int32)
            ), dim=0
        )
        last_input = torch.empty((infer_state.batch_size, self.model_config.hidden_size), device=input_embds.device, dtype=input_embds.dtype)
        last_input[:, :] = input_embds[last_token_indices, :]
        # Apply RMS-norm
        rmsnorm_inplace(
            last_input,
            self.weights.final_norm,
            self.model_config.rms_norm_eps
        )
        logits = linear(last_input, self.weights.lm_head)    # [batch_size, vocab_size]
        output_tokens = torch.argmax(logits, dim=1)
        return output_tokens
    