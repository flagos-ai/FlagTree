# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A stream reduction must satisfy packed consumers at its output store."""

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions import post_function_boundary_pack_propagation
from triton.flagmega.targets import NvidiaSm90Target


def graph(lanes, *, scalar_user=False, output_lanes=()):
    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1, 4, 40)), id="value")
            reduction = fm.F.math.reduce_sum(value, axes=(1,), keep_dims=False, output_lanes=output_lanes,
                                             name="collapse") if output_lanes else fm.F.math.reduce_sum(
                                                 value, axes=(1,), keep_dims=False, name="collapse")
            # AutoPack has already vectorized the mean at this pass boundary.
            vector = fm.F.tensors.pack(reduction, lanes, axes=(-1,) * len(lanes))
            packed = fm.F.math.vectorized_binary(vector, fm.F.builtin.splat_const(vector.type, .25), binary_op="mul")
            outputs = [packed, fm.F.nn.norm_stats(packed, axis=-1, use_mean=False)]
            if scalar_user:
                outputs.append(reduction)
            self.function("main", (value,), tuple(outputs))
    return Graph(dialect="high_level", stage="packed", entry="main").build()


@pytest.mark.parametrize("lanes", [(8,), (2, 4)])
@pytest.mark.parametrize("scalar_user", [False, True])
def test_stream_reduction_produces_packed_output_without_compute_pack(lanes, scalar_user):
    original = graph(lanes, scalar_user=scalar_user)
    rewritten = post_function_boundary_pack_propagation(original, NvidiaSm90Target())
    reduction, = (node for node in rewritten.nodes if node.op == "math.reduce_sum")
    assert reduction.attrs.get("output_lanes") == lanes
    assert not any(node.op == "tensors.pack" for node in rewritten.nodes)
    value = torch.linspace(-5, 3, 160).reshape(1, 4, 40).bfloat16()
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(rewritten, {"value": value}),
                               evaluator.run(original, {"value": value}), rtol=0, atol=0)
    assert post_function_boundary_pack_propagation(rewritten, NvidiaSm90Target()).semantic_hash == rewritten.semantic_hash


def test_reduction_composes_output_lanes_without_recomputing():
    original = graph((4,), output_lanes=(2,), scalar_user=True)
    rewritten = post_function_boundary_pack_propagation(original, NvidiaSm90Target())
    reduction, = (node for node in rewritten.nodes if node.op == "math.reduce_sum")
    assert reduction.attrs["output_lanes"] == (4, 2)
    assert not any(node.op == "tensors.pack" for node in rewritten.nodes)
    value = torch.linspace(-5, 3, 160).reshape(1, 4, 40).bfloat16()
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(rewritten, {"value": value}),
                               evaluator.run(original, {"value": value}), rtol=0, atol=0)
