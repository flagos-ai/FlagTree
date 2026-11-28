# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""QKV matching retains the scalar rotary prefix independently of the head."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.rules.neutral._qkv_head import qkv_head_pattern, qkv_head_from_match
from triton.flagmega.pattern_match import try_match_root


@pytest.mark.parametrize("rotary_dim,matched", [(None, True), (24, True), (16, True)])
def test_qkv_head_match_retains_rotary_contract(rotary_dim, matched):

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1, 2, 24)))
            scale = self.input("scale", fm.tensor_type("bfloat16", (24, )))
            bias = self.input("bias", scale.type)
            table = self.input("table", fm.tensor_type("float32", (1, 1, rotary_dim or 24)))
            stats = fm.F.nn.norm_stats(value, axis=-1, use_mean=False)
            norm = fm.F.nn.norm_apply(value, stats, scale, bias, axis=-1, epsilon=1e-6, use_mean=False, name="norm")
            rope = fm.F.nn.rope(norm, table, table, rotary_dim=rotary_dim, name="rope")
            self.function("main", (value, scale, bias, table), (rope, ))

    module = Graph(dialect="high_level", stage="normalization_decomposed", entry="main").build()
    match = try_match_root(module.node_map["rope"], qkv_head_pattern("q"), module)
    result = None if match is None else qkv_head_from_match(match, "q")
    assert (result is not None) == matched
    assert result.rope.attrs.get("rotary_dim") == rotary_dim
