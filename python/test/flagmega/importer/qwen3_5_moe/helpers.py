# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Small independent mixed-decoder checkpoint; no model downloads in UTs."""

import torch

from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.ir import DType


def configuration(layer_types=("linear_attention", "full_attention", "linear_attention", "full_attention")):
    return {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "tie_word_embeddings": False,
        "text_config": {
            "dtype": "bfloat16",
            "hidden_size": 16,
            "vocab_size": 32,
            "num_hidden_layers": len(layer_types),
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8,
            "layer_types": list(layer_types),
            "rope_parameters": {"rope_type": "default", "partial_rotary_factor": 0.5, "rope_theta": 10000},
            "rms_norm_eps": 1e-6,
        },
    }


def checkpoint(config=None, *, with_values=False):
    config = configuration() if config is None else config
    shapes = {
        "model.language_model.embed_tokens.weight": (32, 16),
        "model.language_model.norm.weight": (16, ),
        "lm_head.weight": (32, 16),
    }
    for layer, kind in enumerate(config["text_config"]["layer_types"]):
        weights = {
            "input_layernorm.weight": (16, ),
            "post_attention_layernorm.weight": (16, ),
            "mlp.gate.weight": (4, 16),
            "mlp.experts.gate_up_proj": (4, 16, 16),
            "mlp.experts.down_proj": (4, 16, 8),
            "mlp.shared_expert.gate_proj.weight": (8, 16),
            "mlp.shared_expert.up_proj.weight": (8, 16),
            "mlp.shared_expert.down_proj.weight": (16, 8),
            "mlp.shared_expert_gate.weight": (1, 16),
        }
        if kind == "linear_attention":
            weights.update({
                "linear_attn.in_proj_qkv.weight": (16, 16),
                "linear_attn.in_proj_z.weight": (8, 16),
                "linear_attn.in_proj_b.weight": (2, 16),
                "linear_attn.in_proj_a.weight": (2, 16),
                "linear_attn.conv1d.weight": (16, 1, 4),
                "linear_attn.A_log": (2, ),
                "linear_attn.dt_bias": (2, ),
                "linear_attn.norm.weight": (4, ),
                "linear_attn.out_proj.weight": (16, 8),
            })
        else:
            weights.update({
                "self_attn.q_proj.weight": (32, 16),
                "self_attn.k_proj.weight": (8, 16),
                "self_attn.v_proj.weight": (8, 16),
                "self_attn.q_norm.weight": (8, ),
                "self_attn.k_norm.weight": (8, ),
                "self_attn.o_proj.weight": (16, 16),
            })
        shapes.update({f"model.language_model.layers.{layer}.{key}": shape for key, shape in weights.items()})
    infos = {
        key:
        TensorInfo(key, DType.FLOAT32 if key.endswith(
            ("linear_attn.A_log", "linear_attn.norm.weight")) else DType.BFLOAT16, shape, "model.safetensors")
        for key, shape in shapes.items()
    }
    generator = torch.Generator().manual_seed(423)
    values = {
        key: (torch.randn(shape, generator=generator) * 0.1).to(getattr(torch, infos[key].dtype.value))
        for key, shape in shapes.items()
    } if with_values else {}
    return MemoryCheckpoint(config, infos, values)
