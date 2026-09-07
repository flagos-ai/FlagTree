# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.norm_stats import NormStats


def _node(name, value_type):
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})


def test_norm_stats_parameter_info_and_dense_type_match_nncase_contract():
    value = _node("value", fm.tensor_type("bfloat16", [3, 5, 7]))
    result = NormStats.infer_type((value,), {"axis": 1, "use_mean": True})

    assert NormStats.value.name == "input"
    assert NormStats.value.type_pattern.reason == "is_tensor"
    assert result == fm.tensor_type("float32", [2, 3, 1, 1])


def test_norm_stats_hidden_split_becomes_additive_partial_on_two_dimensional_mesh():
    placement = fm.Placement((2, 4), "yx", "bb")
    value_type = fm.DistributedType(
        fm.tensor_type("bfloat16", [8, 32]),
        (fm.SBP.split_contiguous((0,)), fm.SBP.split_contiguous((1,))),
        placement,
    )
    result = NormStats.infer_type(
        (_node("value", value_type),), {"axis": 1, "use_mean": False})

    assert isinstance(result, fm.DistributedType)
    assert result.tensor == fm.tensor_type("float32", [1, 8, 1])
    assert result.axis_policies == (
        fm.SBP.broadcast(), fm.SBP.split_contiguous((0,)), fm.SBP.broadcast())
    assert result.partial == fm.SBP.partial((1,))


class _StatsModule(fm.Module):
    def __init__(self, use_mean):
        super().__init__(dialect="nn", stage="imported", entry="main")
        self.use_mean = use_mean

    def forward(self):
        value = self.input("value", fm.tensor_type("float32", [2, 3, 4]))
        stats = fm.F.nn.norm_stats(
            value, axis=1, use_mean=self.use_mean, name="stats")
        self.function("main", (value,), (stats,))


def test_norm_stats_evaluator_returns_sum_then_sum_of_squares():
    module = _StatsModule(True).build()
    value = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 8
    actual = TorchEvaluator(DictWeightResolver({})).run(module, {"value": value})[0]

    expected = torch.cat((
        value.sum(dim=(1, 2), keepdim=True).unsqueeze(0),
        value.square().sum(dim=(1, 2), keepdim=True).unsqueeze(0),
    ), dim=0)
    torch.testing.assert_close(actual, expected)
