# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Validated text-decoder configuration and checkpoint weight contracts."""

from dataclasses import dataclass
from collections.abc import Mapping
import math

from triton.flagmega.errors import ImporterError
from triton.flagmega.ir import tensor_type


@dataclass(frozen=True)
class Qwen35MoeConfig:
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel_size: int
    num_experts: int
    num_experts_per_tok: int
    intermediate_size: int
    shared_intermediate_size: int
    layer_types: tuple[str, ...]
    epsilon: float
    rotary_dim: int
    rope_theta: float
    tie_word_embeddings: bool
    padding_idx: int | None

    @property
    def query_size(self):
        return self.num_attention_heads * self.head_dim

    @property
    def kv_size(self):
        return self.num_key_value_heads * self.head_dim

    @property
    def value_dim(self):
        return self.num_value_heads * self.value_head_dim

    @property
    def conv_dim(self):
        return 2 * self.num_key_heads * self.key_head_dim + self.value_dim

    @classmethod
    def parse(cls, source):
        text = source.get("text_config", source)
        if not isinstance(text, Mapping):
            raise ImporterError("Qwen3.5 MoE text_config must be an object.")
        if source.get("quantization_config") or text.get("quantization_config"):
            raise ImporterError("Qwen3.5 MoE importer currently requires unquantized BF16 weights.")
        if text.get("dtype", text.get("torch_dtype", "bfloat16")) != "bfloat16":
            raise ImporterError("Qwen3.5 MoE importer requires BF16 storage.")
        if text.get("attention_bias", False) or text.get("hidden_act", "silu") != "silu":
            raise ImporterError("Qwen3.5 MoE importer requires bias-free projections and SiLU.")
        if text.get("attn_output_gate", True) is not True or text.get("mlp_only_layers", []):
            raise ImporterError("Qwen3.5 MoE importer requires gated attention and MoE at every decoder layer.")
        if text.get("mamba_ssm_dtype", "float32") != "float32":
            raise ImporterError("Qwen3.5 MoE recurrent state must be FP32.")
        mapping = {
            "hidden_size": "hidden_size",
            "vocab_size": "vocab_size",
            "num_hidden_layers": "num_hidden_layers",
            "num_attention_heads": "num_attention_heads",
            "num_key_value_heads": "num_key_value_heads",
            "head_dim": "head_dim",
            "num_key_heads": "linear_num_key_heads",
            "num_value_heads": "linear_num_value_heads",
            "key_head_dim": "linear_key_head_dim",
            "value_head_dim": "linear_value_head_dim",
            "conv_kernel_size": "linear_conv_kernel_dim",
            "num_experts": "num_experts",
            "num_experts_per_tok": "num_experts_per_tok",
            "intermediate_size": "moe_intermediate_size",
            "shared_intermediate_size": "shared_expert_intermediate_size",
        }
        values = {}
        for name, key in mapping.items():
            value = text.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ImporterError(f"Qwen3.5 MoE {key} must be a positive integer.")
            values[name] = value
        layer_types = text.get("layer_types")
        if layer_types is None:
            interval = text.get("full_attention_interval", 4)
            if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
                raise ImporterError("Qwen3.5 MoE full_attention_interval must be positive.")
            layer_types = tuple("full_attention" if (layer + 1) % interval == 0 else "linear_attention"
                                for layer in range(values["num_hidden_layers"]))
        if (not isinstance(layer_types, (list, tuple)) or len(layer_types) != values["num_hidden_layers"]
                or any(kind not in {"linear_attention", "full_attention"} for kind in layer_types)):
            raise ImporterError("Qwen3.5 MoE layer_types must describe every linear/full attention layer.")
        rope = text.get("rope_parameters", {})
        if not isinstance(rope, Mapping) or rope.get("rope_type", "default") != "default":
            raise ImporterError("Qwen3.5 MoE importer requires default (possibly partial) RoPE.")
        try:
            fraction = float(rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 0.25)))
            theta = float(rope.get("rope_theta", text.get("rope_theta", 10000000)))
            epsilon = float(text.get("rms_norm_eps", 1e-6))
        except (TypeError, ValueError) as error:
            raise ImporterError("Invalid Qwen3.5 MoE RoPE/RMSNorm numeric attributes.") from error
        rotary_dim = values["head_dim"] * fraction
        if not 0 < fraction <= 1 or not rotary_dim.is_integer() or int(rotary_dim) % 2:
            raise ImporterError("Qwen3.5 MoE rotary dimension must be positive, integral and even.")
        if not all(math.isfinite(value) and value > 0 for value in (theta, epsilon)):
            raise ImporterError("Qwen3.5 MoE theta/epsilon must be finite and positive.")
        if values["num_attention_heads"] % values["num_key_value_heads"] or values["num_value_heads"] % values[
                "num_key_heads"]:
            raise ImporterError("Qwen3.5 MoE query/value heads must divide evenly into key-head groups.")
        if values["num_experts_per_tok"] > values["num_experts"]:
            raise ImporterError("Qwen3.5 MoE selected experts cannot exceed the expert bank.")
        tied = text.get("tie_word_embeddings", source.get("tie_word_embeddings", False))
        padding = text.get("pad_token_id", source.get("pad_token_id"))
        if not isinstance(tied, bool) or (padding is not None and
                                          (isinstance(padding, bool) or not isinstance(padding, int)
                                           or not 0 <= padding < values["vocab_size"])):
            raise ImporterError("Qwen3.5 MoE embedding configuration is invalid.")
        return cls(**values, layer_types=tuple(layer_types), epsilon=epsilon, rotary_dim=int(rotary_dim),
                   rope_theta=theta, tie_word_embeddings=tied, padding_idx=padding)

    def weight_types(self, kind):
        h, e, i, shared = self.hidden_size, self.num_experts, self.intermediate_size, self.shared_intermediate_size
        shapes = {
            "input_layernorm.weight": (h, ),
            "post_attention_layernorm.weight": (h, ),
            "mlp.gate.weight": (e, h),
            "mlp.experts.gate_up_proj": (e, 2 * i, h),
            "mlp.experts.down_proj": (e, h, i),
            "mlp.shared_expert.gate_proj.weight": (shared, h),
            "mlp.shared_expert.up_proj.weight": (shared, h),
            "mlp.shared_expert.down_proj.weight": (h, shared),
            "mlp.shared_expert_gate.weight": (1, h),
        }
        if kind == "linear_attention":
            shapes.update({
                "linear_attn.in_proj_qkv.weight": (self.conv_dim, h),
                "linear_attn.in_proj_z.weight": (self.value_dim, h),
                "linear_attn.in_proj_b.weight": (self.num_value_heads, h),
                "linear_attn.in_proj_a.weight": (self.num_value_heads, h),
                "linear_attn.conv1d.weight": (self.conv_dim, 1, self.conv_kernel_size),
                "linear_attn.A_log": (self.num_value_heads, ),
                "linear_attn.dt_bias": (self.num_value_heads, ),
                "linear_attn.norm.weight": (self.value_head_dim, ),
                "linear_attn.out_proj.weight": (h, self.value_dim),
            })
        elif kind == "full_attention":
            shapes.update({
                "self_attn.q_proj.weight": (2 * self.query_size, h),
                "self_attn.k_proj.weight": (self.kv_size, h),
                "self_attn.v_proj.weight": (self.kv_size, h),
                "self_attn.q_norm.weight": (self.head_dim, ),
                "self_attn.k_norm.weight": (self.head_dim, ),
                "self_attn.o_proj.weight": (h, self.query_size),
            })
        else:
            raise ImporterError(f"Unknown Qwen3.5 MoE decoder kind {kind!r}.")
        return {
            name: tensor_type("float32" if name in {"linear_attn.A_log", "linear_attn.norm.weight"} else "bfloat16",
                              shape)
            for name, shape in shapes.items()
        }
