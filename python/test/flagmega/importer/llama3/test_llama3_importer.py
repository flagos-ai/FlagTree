# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Llama 3.x importer contract: scaled RoPE table, no QK-norm, registry."""

from __future__ import annotations

import pytest
import torch

from triton.flagmega.importer import importer_registry
from triton.flagmega.importer.checkpoint import MemoryCheckpoint, TensorInfo
from triton.flagmega.importer.llama3 import (
    Llama3LayerConfig,
    Llama3LayerImporter,
    Llama3ModelImporter,
    llama3_inverse_frequencies,
)
from triton.flagmega.ir import DType


def _config(**overrides):
    config = dict(
        architectures=["LlamaForCausalLM"],
        model_type="llama",
        attention_bias=False,
        mlp_bias=False,
        hidden_act="silu",
        vocab_size=256,
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-05,
        rope_theta=500000.0,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        pad_token_id=None,
        rope_scaling=dict(factor=32.0, high_freq_factor=4.0, low_freq_factor=1.0, original_max_position_embeddings=64,
                          rope_type="llama3"),
    )
    config.update(overrides)
    return config


def _checkpoint(config):
    tensors, values = {}, {}

    def add(key, shape):
        tensors[key] = TensorInfo(key=key, dtype=DType.BFLOAT16, shape=shape, source="synthetic")
        values[key] = torch.randn(shape, dtype=torch.bfloat16)

    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    query = config["num_attention_heads"] * config["head_dim"]
    kv = config["num_key_value_heads"] * config["head_dim"]
    add("model.embed_tokens.weight", (config["vocab_size"], hidden))
    add("model.norm.weight", (hidden, ))
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        add(prefix + "input_layernorm.weight", (hidden, ))
        add(prefix + "self_attn.q_proj.weight", (query, hidden))
        add(prefix + "self_attn.k_proj.weight", (kv, hidden))
        add(prefix + "self_attn.v_proj.weight", (kv, hidden))
        add(prefix + "self_attn.o_proj.weight", (hidden, query))
        add(prefix + "post_attention_layernorm.weight", (hidden, ))
        add(prefix + "mlp.gate_proj.weight", (intermediate, hidden))
        add(prefix + "mlp.up_proj.weight", (intermediate, hidden))
        add(prefix + "mlp.down_proj.weight", (hidden, intermediate))
    return MemoryCheckpoint(config, tensors, values)


def test_registry_resolves_llama():
    spec = importer_registry.resolve(_config(), full_model=True)
    assert spec.name == "llama3"
    assert spec.execution_modes == frozenset({"decode-1", "prefill"})


def test_model_import_has_no_qk_norm_and_pins_scaled_rotary():
    checkpoint = _checkpoint(_config())
    module = Llama3ModelImporter(checkpoint).import_module()
    ops = [node.op for node in module.nodes]
    assert "nn.rotary_embedding" in ops
    assert "nn.rms_norm" in ops
    assert "decode_layer" in module.function_map
    rotary = [node for node in module.nodes if node.op == "nn.rotary_embedding"]
    assert len(rotary) == 1
    attrs = rotary[0].attrs
    assert attrs["head_dim"] == 16
    assert len(attrs["inverse_frequencies"]) == 8

    # The decoder function is built once, so QK-norm nodes would appear
    # exactly once if the shared attention helper still applied them.
    norm_nodes = [
        node for node in module.nodes if node.op == "nn.rms_norm" and node.id.endswith(("query_norm", "key_norm"))
    ]
    assert norm_nodes == []


def test_inverse_frequencies_match_huggingface():
    _transformers = pytest.importorskip("transformers")
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    rope_scaling = dict(factor=32.0, high_freq_factor=4.0, low_freq_factor=1.0, original_max_position_embeddings=8192,
                        rope_type="llama3")
    config = Llama3LayerConfig(
        vocab_size=256,
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        epsilon=1e-5,
        rope_theta=500000.0,
        max_position_embeddings=131072,
        padding_idx=None,
        tie_word_embeddings=True,
        rope_scaling=rope_scaling,
    )
    ours = torch.tensor(llama3_inverse_frequencies(config), dtype=torch.float32)
    reference = LlamaRotaryEmbedding(
        LlamaConfig(head_dim=config.head_dim, hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    max_position_embeddings=config.max_position_embeddings, rope_theta=config.rope_theta,
                    rope_scaling=rope_scaling),
        device="cpu",
    )
    # transformers exposes the computed schedule as a non-persistent buffer.
    expected = reference.inv_freq.detach().float()
    assert torch.equal(ours, expected)


def test_layer_import_requires_single_layer_zero():
    checkpoint = _checkpoint(_config())
    Llama3LayerImporter(checkpoint, layer=0).import_module()
    with pytest.raises(Exception, match="layer 0"):
        Llama3LayerImporter(checkpoint, layer=1)
