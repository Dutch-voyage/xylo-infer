import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from ..layers.activation import SiluAndMul
from ..layers.layernorm import RMSNorm
from ..layers.linear import (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from ..layers.rotary_embedding import get_rope
from ..layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from src.services.nanovllm_v5.engine.sequence import Sequence
from src.artifacts.nanovllm_v5.cache_mngr.layerwise import CacheManager

from src.core.service_base import BaseService
from src.core.artifact_base import Artifact
import dataclasses

from src.services.nanovllm_v5.utils.context import get_cuda_graph_flag

from src.artifacts.nanovllm_v5.attention.flashinfer_attention import Attention


@dataclasses.dataclass
class Qwen3AttentionArtifacts:
    attention: Artifact

    @classmethod
    def init_new(
        cls,
        model_runner, 
        config
    ):
        
        tp_size = dist.get_world_size()
        num_heads = config.num_attention_heads // tp_size
        num_kv_heads = config.num_key_value_heads // tp_size
        head_dim = config.head_dim
        scaling = head_dim**-0.5
        return cls(
            attention=Attention(
                model_runner, 
                num_heads,
                head_dim,
                scaling,
                num_kv_heads,
            )
        )

    def register(self, service: BaseService):
        if "attention" in service.name.lower():
            self.attention.register_for_attn(service)
        # if "runner" in service.name.lower():    
        #     # self.attention._register_method("init_forward_metadata_capture_cuda_graph", service)
        #     # self.attention._register_method("init_forward_metadata_replay_cuda_graph", service)
        #     # self.attention._register_method("update_indices", service)
        #     self.attention.register_for_runner(service)
        if "cachemanager" in service.name.lower():
            self.attention._register_method("prepare_metadata_for_attn", service)
            self.attention._register_method("init_forward_metadata_capture_cuda_graph", service)
            self.attention._register_method("init_forward_metadata_replay_cuda_graph", service)
            self.attention._register_obj("decode_cuda_graph_metadata", service)


class Qwen3Attention(nn.Module, BaseService):
    @property
    def name(self):
        return f"Qwen3Attention_layer_{self.layer_id}"
    
    def __init__(
        self,
        layer_id: int, 
        attention_backend: any, 
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        BaseService.__init__(self)
        self.layer_id = layer_id
        
        self.k_cache = self.v_cache = self.q_cache = torch.tensor([])
        
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        
        attention_backend.register(self)
                
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        # o = self.attn(q, k, v, self.layer_id)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        layer_id: int, 
        attention_backend, 
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = Qwen3Attention(
            layer_id, 
            attention_backend,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        attention_backend, 
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        
        
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(layer_id, attention_backend, config) for layer_id in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, attention_backend, config: Qwen3Config) -> None:
        super().__init__()
        self.model = Qwen3Model(attention_backend, config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits
