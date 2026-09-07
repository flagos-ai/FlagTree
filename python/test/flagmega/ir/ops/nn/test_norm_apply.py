# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.norm_apply import NormApply
from triton.flagmega.ir.ops.nn.norm_stats import NormStats


def _node(name, value_type):
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})


@pytest.mark.parametrize("use_mean", [False, True])
def test_norm_apply_dense_type_and_evaluator(use_mean):
    class _Module(fm.Module):
        def __init__(self):
            super().__init__(dialect="nn", stage="imported", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", [2, 8]))
            scale = self.input("scale", fm.tensor_type("bfloat16", [8]))
            bias = self.input("bias", fm.tensor_type("bfloat16", [8]))
            stats = fm.F.nn.norm_stats(
                value, axis=-1, use_mean=use_mean, name="stats")
            output = fm.F.nn.norm_apply(
                value,
                stats,
                scale,
                bias,
                axis=-1,
                epsilon=1e-6,
                use_mean=use_mean,
                name="output",
            )
            self.function("main", (value, scale, bias), (output,))

    module = _Module().build()
    value = torch.randn((2, 8), dtype=torch.bfloat16)
    scale = torch.randn((8,), dtype=torch.bfloat16)
    bias = torch.randn((8,), dtype=torch.bfloat16)
    actual = TorchEvaluator(DictWeightResolver({})).run(
        module, {"value": value, "scale": scale, "bias": bias})[0]
    source = value.float()
    if use_mean:
        mean = source.mean(dim=-1, keepdim=True)
        variance = source.square().mean(dim=-1, keepdim=True) - mean.square()
        normalized = (source - mean) * (variance.clamp_min(0) + 1e-6).rsqrt()
    else:
        normalized = source * (source.square().mean(dim=-1, keepdim=True) + 1e-6).rsqrt()
    expected = (normalized * scale.float() + bias.float()).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected)


def test_norm_apply_accepts_materialized_stats_and_matching_suffix_split():
    placement = fm.Placement((8,), "x", "b")
    value_tensor = fm.tensor_type("bfloat16", [2, 64])
    parameter_tensor = fm.tensor_type("bfloat16", [64])
    value_type = fm.DistributedType(
        value_tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,))),
        placement,
    )
    parameter_type = fm.DistributedType(
        parameter_tensor, (fm.SBP.split_contiguous((0,)),), placement)
    value = _node("value", value_type)
    partial = NormStats.infer_type((value,), {"axis": 1, "use_mean": False})
    assert isinstance(partial, fm.DistributedType)
    materialized = fm.DistributedType(
        partial.tensor, partial.axis_policies, partial.placement)

    result = NormApply.infer_type(
        (
            value,
            _node("stats", materialized),
            _node("scale", parameter_type),
            _node("bias", parameter_type),
        ),
        {"axis": 1, "epsilon": 1e-6, "use_mean": False},
    )
    assert result == value_type


def test_norm_apply_rejects_stats_shape_and_parameter_policy_mismatch():
    value = _node("value", fm.tensor_type("float32", [2, 8]))
    scale = _node("scale", fm.tensor_type("float32", [8]))
    bias = _node("bias", fm.tensor_type("float32", [8]))
    with pytest.raises(IRSchemaError, match="stats type"):
        NormApply.infer_type(
            (value, _node("stats", fm.tensor_type("float32", [1, 2, 1, 1])), scale, bias),
            {"axis": 1, "epsilon": 1e-6, "use_mean": False},
        )
