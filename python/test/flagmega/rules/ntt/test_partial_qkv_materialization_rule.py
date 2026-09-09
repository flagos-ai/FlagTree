# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.rules.ntt.fuse_gather_reduce_qkv_rope_with_cache import fuse_gather_reduce_qkv_rope_with_cache_rule
from python.test.flagmega.passes.tir.test_fuse_gather_reduce_qkv_rope_with_cache import _graph


def test_pattern_fuses_view_chains_and_retains_partial_rotary_extent(tmp_path):
    source = _graph(rotary_dim=2)
    rule = fuse_gather_reduce_qkv_rope_with_cache_rule()
    assert rule.pattern is not None and rule.matches is None
    result = DataflowPass("fusion", (rule, ), rewrite_constants=False).run(source)
    fused = result.node_map["qkv_rope"]
    assert fused.op == "ntt.gather_reduce_qkv_rope_with_cache"
    assert fused.attrs["rotary_dim"] == 2
    assert fused.type == source.node_map["qkv_rope"].type
    assert fm.load_module(fm.emit_module(result, tmp_path / "fused.py")) == result


def test_pattern_does_not_consume_a_shared_view():
    source = _graph(extra_q_user=True, rotary_dim=2)
    assert fuse_gather_reduce_qkv_rope_with_cache_rule().apply(source.node_map["qkv_rope"], source) is None
