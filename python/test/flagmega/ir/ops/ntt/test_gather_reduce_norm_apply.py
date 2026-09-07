# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.ntt.gather_reduce_norm_apply import (
    GatherReduceNormApply,
)


def _types(reduce_op=fm.ReduceOp.SUM):
    placement = fm.Placement((2, 4), "yx", "bb")
    broadcast = fm.SBP.broadcast()
    split = fm.SBP.split_contiguous((0, 1))
    value = fm.DistributedType(
        fm.tensor_type("bfloat16", (1, 8)),
        (broadcast, split),
        placement,
    )
    parameter = fm.DistributedType(
        fm.tensor_type("bfloat16", (8,)), (split,), placement
    )
    materialized = fm.DistributedType(
        fm.tensor_type("float32", (1, 1, 1)),
        (broadcast, broadcast, broadcast),
        placement,
    )
    partial = fm.DistributedType(
        materialized.tensor,
        materialized.axis_policies,
        placement,
        fm.SBP.partial((0, 1), reduce_op),
    )
    return value, parameter, partial, materialized


def _node(name, value_type):
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})


def test_type_requires_sum_partial_statistics_and_preserves_value_layout():
    value_type, parameter_type, partial_type, materialized_type = _types()
    result = GatherReduceNormApply.infer_type(
        (
            _node("partial", partial_type),
            _node("value", value_type),
            _node("scale", parameter_type),
            _node("bias", parameter_type),
        ),
        {
            "materialized_stats_type": materialized_type,
            "axis": -1,
            "epsilon": 1e-6,
            "use_mean": False,
            "has_bias": True,
        },
    )
    assert result == value_type
    assert GatherReduceNormApply.partial_stats.name == "partial_stats"
    assert GatherReduceNormApply.value.name == "input"


def test_type_rejects_non_sum_partial_statistics():
    value_type, parameter_type, partial_type, materialized_type = _types(
        fm.ReduceOp.MAX
    )
    with pytest.raises(IRSchemaError, match="Sum-partial"):
        GatherReduceNormApply.infer_type(
            (
                _node("partial", partial_type),
                _node("value", value_type),
                _node("scale", parameter_type),
                _node("bias", parameter_type),
            ),
            {
                "materialized_stats_type": materialized_type,
                "axis": -1,
                "epsilon": 1e-6,
                "use_mean": False,
                "has_bias": True,
            },
        )


def test_evaluator_uses_logical_partial_value_and_can_elide_proven_zero_bias():
    value_type, parameter_type, partial_type, materialized_type = _types()

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="distributed", entry="main")

        def forward(self):
            partial = self.input("partial", partial_type, id="partial")
            value = self.input("value", value_type, id="value")
            scale = self.input("scale", parameter_type, id="scale")
            bias = self.input("bias", parameter_type, id="bias")
            output = fm.F.ntt.gather_reduce_norm_apply(
                partial,
                value,
                scale,
                bias,
                materialized_stats_type=materialized_type,
                axis=-1,
                epsilon=1e-6,
                use_mean=False,
                has_bias=False,
                name="output",
            )
            self.function("main", (partial, value, scale, bias), (output,))

    value = torch.randn((1, 8), dtype=torch.bfloat16)
    scale = torch.randn((8,), dtype=torch.bfloat16)
    bias = torch.full((8,), 100, dtype=torch.bfloat16)
    stats = value.float().square().sum(-1, keepdim=True).unsqueeze(0)
    actual = TorchEvaluator(DictWeightResolver({})).run(
        Graph().build(),
        {"partial": stats, "value": value, "scale": scale, "bias": bias},
    )[0]
    expected = (
        value.float()
        * torch.rsqrt(stats.squeeze(0) / 8 + 1e-6)
        * scale.float()
    ).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected)
