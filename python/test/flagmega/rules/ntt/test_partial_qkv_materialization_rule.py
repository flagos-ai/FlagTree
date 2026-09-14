# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.tir.fuse_distributed_ops import fuse_distributed_ops
from python.test.flagmega.passes.tir.test_explicit_qkv_materialization import _graph


def test_explicit_collective_retains_partial_rotary_extent(tmp_path):
    source = _graph(rotary_dim=2)
    result = fuse_distributed_ops(source)
    fused = result.node_map["qkv_rope"]
    assert fused.op == "nn.qkv_rope_with_cache"
    assert result.node_map["combined"].op == "distributed.boxing"
    assert fused.attrs["rotary_dim"] == 2
    assert fused.type == source.node_map["qkv_rope"].type
    assert fm.load_module(fm.emit_module(result, tmp_path / "fused.py")) == result


def test_shared_view_retains_its_explicit_collective():
    source = _graph(extra_q_user=True, rotary_dim=2)
    result = fuse_distributed_ops(source)
    assert result.node_map["qkv_rope"].op == "nn.qkv_rope_with_cache"
    assert result.node_map["combined"].op == "distributed.boxing"
