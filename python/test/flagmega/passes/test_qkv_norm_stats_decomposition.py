# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Q/K reductions must be visible to distribution, not hidden in RoPE/cache."""

from triton.flagmega.compiler import Compiler
from python.test.flagmega.passes.target_independent.test_form_qkv_rope_with_cache import QKVRoPERegion


def test_qkv_normalization_reductions_are_explicit_before_distribution():
    original = QKVRoPERegion().build()
    result = Compiler().compile(original, stop_after="decompose-gdn").module
    stats = [node for node in result.nodes if node.op == "nn.norm_stats"]
    assert len(stats) == 2
    apply, = (node for node in result.nodes if node.op == "nn.qkv_rope_with_cache")
    assert len(apply.inputs) == 12
    assert {node.id for node in stats}.issubset(apply.inputs)
    assert result.node_map["attention"].type == original.node_map["attention"].type
