# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.ir import DType


PREFIX = "model.layers.0."


def config() -> dict[str, object]:
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": 32,
        "num_hidden_layers": 1,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 128,
        "tie_word_embeddings": True,
        "pad_token_id": None,
    }


def tensor_specs() -> dict[str, tuple[int, ...]]:
    return {
        "model.embed_tokens.weight": (32, 16),
        PREFIX + "input_layernorm.weight": (16,),
        PREFIX + "self_attn.q_proj.weight": (16, 16),
        PREFIX + "self_attn.k_proj.weight": (8, 16),
        PREFIX + "self_attn.v_proj.weight": (8, 16),
        PREFIX + "self_attn.q_norm.weight": (8,),
        PREFIX + "self_attn.k_norm.weight": (8,),
        PREFIX + "self_attn.o_proj.weight": (16, 16),
        PREFIX + "post_attention_layernorm.weight": (16,),
        PREFIX + "mlp.gate_proj.weight": (32, 16),
        PREFIX + "mlp.up_proj.weight": (32, 16),
        PREFIX + "mlp.down_proj.weight": (16, 32),
        "model.norm.weight": (16,),
    }


def checkpoint(seed: int = 1234) -> MemoryCheckpoint:
    generator = torch.Generator().manual_seed(seed)
    infos = {}
    values = {}
    for key, shape in tensor_specs().items():
        infos[key] = TensorInfo(key, DType.BFLOAT16, shape, "qwen3-unit.safetensors")
        if key.endswith("norm.weight") or key.endswith("layernorm.weight"):
            value = torch.ones(shape, dtype=torch.bfloat16)
        else:
            value = (torch.randn(shape, generator=generator) * 0.05).to(torch.bfloat16)
        values[key] = value
    return MemoryCheckpoint(config(), infos, values)


def full_checkpoint(*, num_hidden_layers: int = 2, seed: int = 1234) -> MemoryCheckpoint:
    model_config = {**config(), "num_hidden_layers": num_hidden_layers}
    specs = {
        "model.embed_tokens.weight": (32, 16),
        "model.norm.weight": (16,),
    }
    for layer in range(num_hidden_layers):
        prefix = f"model.layers.{layer}."
        specs.update({
            prefix + "input_layernorm.weight": (16,),
            prefix + "self_attn.q_proj.weight": (16, 16),
            prefix + "self_attn.k_proj.weight": (8, 16),
            prefix + "self_attn.v_proj.weight": (8, 16),
            prefix + "self_attn.q_norm.weight": (8,),
            prefix + "self_attn.k_norm.weight": (8,),
            prefix + "self_attn.o_proj.weight": (16, 16),
            prefix + "post_attention_layernorm.weight": (16,),
            prefix + "mlp.gate_proj.weight": (32, 16),
            prefix + "mlp.up_proj.weight": (32, 16),
            prefix + "mlp.down_proj.weight": (16, 32),
        })
    generator = torch.Generator().manual_seed(seed)
    infos = {}
    values = {}
    for key, shape in specs.items():
        infos[key] = TensorInfo(key, DType.BFLOAT16, shape, "qwen3-full-unit.safetensors")
        if key.endswith("norm.weight") or key.endswith("layernorm.weight"):
            value = torch.ones(shape, dtype=torch.bfloat16)
        else:
            value = (torch.randn(shape, generator=generator) * 0.05).to(torch.bfloat16)
        values[key] = value
    return MemoryCheckpoint(model_config, infos, values)


def full_metadata_checkpoint(*, num_hidden_layers: int = 2) -> MemoryCheckpoint:
    """Shape-realistic checkpoint metadata without allocating model weights."""

    hidden = 2048
    intermediate = 6144
    vocab = 4096
    model_config = {
        **config(),
        "vocab_size": vocab,
        "num_hidden_layers": num_hidden_layers,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    specs = {
        "model.embed_tokens.weight": (vocab, hidden),
        "model.norm.weight": (hidden,),
    }
    for layer in range(num_hidden_layers):
        prefix = f"model.layers.{layer}."
        specs.update({
            prefix + "input_layernorm.weight": (hidden,),
            prefix + "self_attn.q_proj.weight": (hidden, hidden),
            prefix + "self_attn.k_proj.weight": (1024, hidden),
            prefix + "self_attn.v_proj.weight": (1024, hidden),
            prefix + "self_attn.q_norm.weight": (128,),
            prefix + "self_attn.k_norm.weight": (128,),
            prefix + "self_attn.o_proj.weight": (hidden, hidden),
            prefix + "post_attention_layernorm.weight": (hidden,),
            prefix + "mlp.gate_proj.weight": (intermediate, hidden),
            prefix + "mlp.up_proj.weight": (intermediate, hidden),
            prefix + "mlp.down_proj.weight": (hidden, intermediate),
        })
    infos = {
        key: TensorInfo(key, DType.BFLOAT16, shape, "qwen3-metadata.safetensors")
        for key, shape in specs.items()
    }
    return MemoryCheckpoint(model_config, infos)
