# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.nn.rms_norm import RMSNorm


def test_rms_norm_infers_broadcast_distributed_result():
    placement = fm.Placement((8,), "b", "b")
    value = fm.tensor_type("bfloat16", [1, 2048])
    weight = fm.tensor_type("bfloat16", [2048])
    value_dist = fm.DistributedType(
        value, (fm.SBP.broadcast(), fm.SBP.broadcast()), placement)
    weight_dist = fm.DistributedType(weight, (fm.SBP.broadcast(),), placement)
    inputs = (
        fm.Node("value", "builtin.var", (), value_dist, attrs={"name": "value"}),
        fm.Node("weight", "builtin.var", (), weight_dist, attrs={"name": "weight"}),
    )

    assert RMSNorm.infer_type(inputs, {"epsilon": 1e-6, "weight_bias": 0.0}) == value_dist


def test_rms_norm_preserves_non_reduction_axis_distribution():
    placement = fm.Placement((8, 16), "yx", "bb")
    value = fm.tensor_type("bfloat16", [1, 16, 128])
    weight = fm.tensor_type("bfloat16", [128])
    value_dist = fm.DistributedType(
        value,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 1), fm.SBP.broadcast()),
        placement,
    )
    weight_dist = fm.DistributedType(
        weight, (fm.SBP.broadcast(),), placement)
    inputs = (
        fm.Node("value", "builtin.var", (), value_dist, attrs={"name": "value"}),
        fm.Node("weight", "builtin.var", (), weight_dist, attrs={"name": "weight"}),
    )

    assert RMSNorm.infer_type(
        inputs, {"epsilon": 1e-6, "weight_bias": 0.0}
    ) == value_dist


def test_rms_norm_rejects_hidden_split_without_stats_decomposition():
    placement = fm.Placement((8,), "b", "b")
    value = fm.tensor_type("bfloat16", [1, 16, 128])
    weight = fm.tensor_type("bfloat16", [128])
    value_dist = fm.DistributedType(
        value,
        (fm.SBP.broadcast(), fm.SBP.broadcast(), fm.SBP.split_contiguous((0,))),
        placement,
    )
    weight_dist = fm.DistributedType(
        weight, (fm.SBP.split_contiguous((0,)),), placement)
    inputs = (
        fm.Node("value", "builtin.var", (), value_dist, attrs={"name": "value"}),
        fm.Node("weight", "builtin.var", (), weight_dist, attrs={"name": "weight"}),
    )

    with pytest.raises(IRSchemaError, match="reduction axis"):
        RMSNorm.infer_type(inputs, {"epsilon": 1e-6, "weight_bias": 0.0})


def test_partial_broadcast_policies_are_not_treated_as_replicated():
    placement = fm.Placement((8,), "b", "b")
    tensor = fm.tensor_type("bfloat16", [1, 2048])
    partial = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
        partial=fm.SBP.partial((0,)),
    )

    from triton.flagmega.ir.distributed_inference import all_broadcast

    assert not all_broadcast(partial)
