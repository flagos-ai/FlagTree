# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch
import pytest

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions import form_add_norm_stats


class _ProjectionResidualStats(fm.Module):
    def __init__(self, *, axis=-1, conflicting_stats=False, wide=False):
        super().__init__(dialect="ntt", stage="stats_threaded", entry="main")
        self.axis = axis
        self.conflicting_stats = conflicting_stats
        self.wide = wide

    def forward(self):
        dtype = "bfloat16" if self.wide else "float32"
        lhs = self.input("lhs", fm.tensor_type(dtype, (2, 8)), id="lhs")
        rhs = self.input("rhs", fm.tensor_type(dtype, (4, 8)), id="rhs")
        residual = self.input(
            "residual", fm.tensor_type("float32", (2, 4)), id="residual")
        projection = fm.F.math.matmul(
            lhs, rhs, transpose_b=True, name="projection")
        if self.wide:
            projection = fm.F.tensors.cast(projection, dtype="float32", name="promotion")
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
    rewritten = form_add_norm_stats(original)

    combines = [
        node for node in rewritten.nodes
        if node.op == "ntt.add_norm_stats"
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

    assert form_add_norm_stats(non_last) == non_last
    assert form_add_norm_stats(conflicting) == conflicting


def test_forms_combine_over_explicit_promotion_without_removing_projection_rounding():
    source = _ProjectionResidualStats(wide=True).build()
    result = form_add_norm_stats(source)
    combine = [node for node in result.nodes if node.op == "ntt.add_norm_stats"]
    assert len(combine) == 1
    assert combine[0].inputs == ("promotion", "residual")
    assert result.node_map["promotion"] == source.node_map["promotion"]
    values = {"lhs": torch.randn(2, 8).bfloat16(), "rhs": torch.randn(4, 8).bfloat16(),
              "residual": torch.randn(2, 4)}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    for actual, expected in zip(evaluator.run(result, values), evaluator.run(source, values), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("vectorized", [False, True])
def test_forms_generic_add_stats_before_distribution_without_a_matmul(vectorized):
    class Graph(fm.Module):
        def forward(self):
            dtype = fm.vector_type("bfloat16", (8,)) if vectorized else "bfloat16"
            value_type = fm.tensor_type(dtype, (1, 4))
            first = self.input("first", value_type, id="first")
            second = self.input("second", value_type, id="second")
            value = (fm.F.math.vectorized_binary(first, second, binary_op="add", name="value")
                     if vectorized else fm.F.math.add(first, second, name="value"))
            stats = fm.F.nn.norm_stats(value, axis=1, use_mean=False, name="stats")
            self.function("main", (first, second), (value, stats))

    source = Graph(dialect="ntt", stage="stats_threaded", entry="main").build()
    result = form_add_norm_stats(source)
    combines = [node for node in result.nodes if node.op == "ntt.add_norm_stats"]
    assert len(combines) == 1
    assert combines[0].inputs == ("first", "second")
    assert "matmul_producer" not in combines[0].metadata
    assert result.node_map["value"].op == result.node_map["stats"].op == "builtin.get_item"
    shape = (1, 4, 8) if vectorized else (1, 4)
    inputs = {name: torch.randn(shape).bfloat16() for name in ("first", "second")}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    for actual, expected in zip(evaluator.run(result, inputs), evaluator.run(source, inputs), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
