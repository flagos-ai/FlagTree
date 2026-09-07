# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions import form_matmul_norm_stats_combine


class _ProjectionResidualStats(fm.Module):
    def __init__(self, *, axis=-1, conflicting_stats=False):
        super().__init__(dialect="ntt", stage="stats_threaded", entry="main")
        self.axis = axis
        self.conflicting_stats = conflicting_stats

    def forward(self):
        lhs = self.input("lhs", fm.tensor_type("float32", (2, 8)), id="lhs")
        rhs = self.input("rhs", fm.tensor_type("float32", (4, 8)), id="rhs")
        residual = self.input(
            "residual", fm.tensor_type("float32", (2, 4)), id="residual")
        projection = fm.F.math.matmul(
            lhs, rhs, transpose_b=True, name="projection")
        value = fm.F.math.add(residual, projection, name="value")
        stats = fm.F.nn.norm_stats(
            value, axis=self.axis, use_mean=False, name="stats")
        outputs = [value, stats]
        if self.conflicting_stats:
            outputs.append(fm.F.nn.norm_stats(
                value, axis=self.axis, use_mean=True, name="mean_stats"))
        self.function("main", (lhs, rhs, residual), outputs)


def test_forms_one_explicit_combine_and_redirects_value_and_stats():
    original = _ProjectionResidualStats().build()
    rewritten = form_matmul_norm_stats_combine(original)

    combines = [
        node for node in rewritten.nodes
        if node.op == "ntt.matmul_norm_stats_combine"
    ]
    assert len(combines) == 1
    assert combines[0].inputs == ("projection", "residual")
    assert rewritten.node_map["value"].op == "builtin.get_item"
    assert rewritten.node_map["value"].inputs == (combines[0].id,)
    assert rewritten.node_map["value"].attrs["index"] == 0
    assert rewritten.node_map["stats"].op == "builtin.get_item"
    assert rewritten.node_map["stats"].inputs == (combines[0].id,)
    assert rewritten.node_map["stats"].attrs["index"] == 1
    assert rewritten.function_map["main"].outputs == ("value", "stats")

    feeds = {
        "lhs": torch.randn(2, 8),
        "rhs": torch.randn(4, 8),
        "residual": torch.randn(2, 4),
    }
    evaluator = TorchEvaluator(DictWeightResolver({}))
    expected = evaluator.run(original, feeds)
    actual = evaluator.run(rewritten, feeds)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_skips_non_last_axis_and_conflicting_stats_contracts():
    non_last = _ProjectionResidualStats(axis=0).build()
    conflicting = _ProjectionResidualStats(conflicting_stats=True).build()

    assert form_matmul_norm_stats_combine(non_last) == non_last
    assert form_matmul_norm_stats_combine(conflicting) == conflicting
