# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""QKV's input-precision contract cannot absorb a distinct norm conversion."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.rules.neutral._qkv_head import qkv_head_pattern, qkv_head_from_match
from triton.flagmega.pattern_match import try_match_root


@pytest.mark.parametrize("source,target", [("bfloat16", "float32"), ("float32", "bfloat16")])
def test_head_match_preserves_normalization_output_rounding(source, target):
    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type(source, (1, 2, 64)))
            scale = self.input("scale", fm.tensor_type(source, (64,)))
            bias = self.input("bias", fm.tensor_type(source, (64,)))
            cos = self.input("cos", fm.tensor_type(target, (1, 1, 64)))
            sin = self.input("sin", cos.type)
            stats = fm.F.nn.norm_stats(x, axis=-1, use_mean=False)
            norm = fm.F.nn.norm_apply(x, stats, scale, bias, axis=-1, epsilon=1e-6,
                use_mean=False, output_dtype=target, name="norm")
            rope = fm.F.nn.rope(norm, cos, sin, name="rope")
            self.function("main", (x, scale, bias, cos, sin), (rope,))

    module = Graph(dialect="high_level", stage="decomposed", entry="main").build()
    match = try_match_root(module.node_map["rope"], qkv_head_pattern("q"), module)
    assert match is None or qkv_head_from_match(match, "q") is None
