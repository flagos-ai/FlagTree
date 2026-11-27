# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Relaxed cast folding must not hide the wide QKV/RoPE fusion region."""

import pytest

from triton.flagmega.rules import DataflowRewriter
from triton.flagmega.rules.neutral.fold_cast import fold_cast_rule
from triton.flagmega.rules.neutral.form_qkv_rope_with_cache import form_qkv_rope_with_cache_rule
from python.test.flagmega.passes.target_independent.test_form_qkv_rope_with_cache import QKVRoPERegion


@pytest.mark.parametrize("rotary_dim", [None, 32])
def test_wide_qkv_with_direct_float_tables_fuses_after_round_trip_elision(rotary_dim):
    original = QKVRoPERegion(wide=True, rotary_dim=rotary_dim).build()
    simplified = DataflowRewriter((fold_cast_rule(), )).rewrite(original)
    assert "cos_wide" not in simplified.node_map
    fused = DataflowRewriter((form_qkv_rope_with_cache_rule(), )).rewrite(simplified)
    qkv, = (node for node in fused.nodes if node.op == "nn.qkv_rope_with_cache")
    assert qkv.attrs["round_qk_intermediates"] is False
    assert fused.node_map[qkv.inputs[5]].type.dtype.value == "float32"
    assert fused.node_map[qkv.inputs[6]].type.dtype.value == "float32"
