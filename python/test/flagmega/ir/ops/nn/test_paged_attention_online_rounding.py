# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Retain the native unnormalized BF16 probability store before PV."""

import json
from pathlib import Path

import torch
import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.nn.paged_attention import PagedAttention
from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state, TorchEvaluator, DictWeightResolver
from python.test.flagmega.ir.ops.primitive_helpers import primitive_module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_flash_attention_probability_rounding():
    fixture = json.loads((Path(__file__).resolve().parents[3] / "data" /
                          "native_flash_attention_online_bf16.json").read_text())
    q, k, v, expected = (torch.tensor(fixture[n], dtype=torch.bfloat16, device="cuda")
                          for n in ("query", "key", "value", "output"))
    config = PagedAttentionStateConfig(1, 1, 256, block_size=64, num_blocks=1)
    state = create_paged_attention_state(config, device="cuda")
    for row in range(33):
        state.kv_caches[0, 0, 0, row].copy_(k[row].reshape(1, 32, 8))
        state.kv_caches[0, 0, 1, row].copy_(v[row].reshape(1, 32, 8))
    state.slot_mapping.fill_(32)
    state.seq_lens.fill_(33)
    module = primitive_module(PagedAttention, (fm.tensor_type("bfloat16", (1, 1, 256)),
                                               config.ref_type, fm.tensor_type("int32", ())),
                       scale=1/16, layout=("seq", "head", "dim"), hidden_size=256,
                       probability_rounding="online_bf16", softmax_block_size=64)
    actual = TorchEvaluator(DictWeightResolver({})).run(module, {
        "q": q, "state": state, "layer_id": torch.tensor(0, dtype=torch.int32, device="cuda")})[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fixture_name", ["native_flash_attention_short_causal_tile.json",
                                          "native_flash_attention_fma_reciprocal.json"])
def test_native_short_causal_query_retains_mma_tile(fixture_name):
    fixture = json.loads((Path(__file__).resolve().parents[3] / "data" /
                          fixture_name).read_text())
    q, k, v, expected = (torch.tensor(fixture[n], dtype=torch.bfloat16, device="cuda")
                        for n in ("query", "key", "value", "output"))
    config = PagedAttentionStateConfig(1, 2, 256, block_size=64, num_blocks=1)
    state = create_paged_attention_state(config, device="cuda")
    length = k.shape[0]
    state.kv_caches[0, 0, 0, :length].copy_(k.reshape(length, 2, 32, 8))
    state.kv_caches[0, 0, 1, :length].copy_(v.reshape(length, 2, 32, 8))
    state.slot_mapping.fill_(length - 1)
    state.seq_lens.fill_(length)
    module = primitive_module(PagedAttention, (
        fm.tensor_type("bfloat16", (1, 16, 256)), config.ref_type, fm.tensor_type("int32", ())),
        scale=1/16, layout=("seq", "head", "dim"), hidden_size=4096,
        probability_rounding="online_bf16", softmax_block_size=80, softmax_traversal="reverse")
    actual = TorchEvaluator(DictWeightResolver({})).run(module, {
        "q": q, "state": state, "layer_id": torch.tensor(0, dtype=torch.int32, device="cuda")})[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
