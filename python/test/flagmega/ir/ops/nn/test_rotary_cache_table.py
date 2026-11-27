# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Readonly rotary recipes retain the native cache bytes and input ABI."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from triton.flagmega import ir as fm, pattern_match as pm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import (
    DictWeightResolver, PagedAttentionStateConfig, TorchEvaluator, create_paged_attention_state,
)


def test_native_rotary_recipe_bytes_and_round_trip(tmp_path):
    fixture = json.loads((Path(__file__).resolve().parents[3] / "data" /
                          "native_rotary_cache.json").read_text())

    class Cache(fm.Module):
        def forward(self):
            table = fm.F.nn.rotary_cache_table(positions=fixture["tokens"],
                inverse_frequencies=tuple(fixture["inverse_frequencies"]))
            self.function("main", (), (table,))

    module = Cache(dialect="high_level", stage="imported", entry="main").build()
    fm.emit_module(module, tmp_path / "cache.py")
    restored = fm.load_module(tmp_path / "cache.py")
    assert module.semantic_hash == restored.semantic_hash
    table, = TorchEvaluator(DictWeightResolver({})).run(restored, {})
    half = fixture["head_dim"] // 2
    packed = torch.cat((table[:, 0, :half], table[:, 1, :half]), -1)
    digest = hashlib.sha256(packed.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert digest == fixture["cache_sha256"]


def test_cached_rotary_position_and_variadic_pattern():
    config = PagedAttentionStateConfig(1, 2, 8, block_size=4, num_blocks=2)

    class Cached(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (1, 17)))
            state = self.input("state", config.ref_type)
            cache = fm.F.nn.rotary_cache_table(positions=8, inverse_frequencies=(1., .1, .01, .001), name="cache")
            pair = fm.F.nn.rotary_embedding(x, state, cache, head_dim=8,
                                            theta=10000., output_dtype="bfloat16", name="rotary")
            self.function("main", (x, state), (*fm.F.tensors.get_items(pair, 0, 1), cache))

    module = Cached(dialect="high_level", stage="imported", entry="main").build()
    state = create_paged_attention_state(config)
    state.seq_lens.fill_(3)
    cos, sin, table = TorchEvaluator(DictWeightResolver({})).run(module, {
        "x": torch.zeros((1, 17), dtype=torch.bfloat16), "state": state})
    assert torch.equal(cos[0, 0], table[3, 0])
    assert torch.equal(sin[0, 0], table[3, 1])
    assert pm.try_match_root(module.node_map["rotary"],
        pm.F.nn.is_rotary_embedding(head_dim=8), module) is not None


@pytest.mark.parametrize("attrs", [
    {"positions": 0}, {"positions": True}, {"inverse_frequencies": ()},
    {"inverse_frequencies": (float("inf"),)}, {"inverse_frequencies": (-1.,)},
    {"positions": 105615}, {"attention_scaling": 0.}, {"output_dtype": "float16"},
])
def test_rotary_recipe_rejects_invalid_contract(attrs):
    with pytest.raises(IRSchemaError):
        fm.get_definition("nn.rotary_cache_table").normalize_attrs({
            "positions": 8, "inverse_frequencies": (1.,), **attrs})
