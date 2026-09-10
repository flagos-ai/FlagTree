# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Late fusion invalidates the replaced calls' materialized choices atomically."""

from dataclasses import replace

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.fusion import fusion_rules
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.rewriter import DataflowPass


def _graph(mode, exported=False):

    class Graph(fm.Module):

        def forward(self):
            value_type = fm.DistributedType(
                fm.tensor_type(fm.vector_type("bfloat16", (8, )), (2, 4)),
                (fm.SBP.split_contiguous((0, )), fm.SBP.broadcast()),
                fm.Placement((2, 2), "xy", "bb"),
            )
            x = self.input("x", value_type, id="x")
            if mode in {"pre", "pre_repeated"}:
                producer = fm.F.ntt.vectorized_cast(x, fm.vector_type("float32", (2, 4)), (1, ), name="producer")
                root = (fm.F.math.vectorized_binary(producer, producer, binary_op="mul", name="root") if mode
                        == "pre_repeated" else fm.F.math.vectorized_unary(producer, unary_op="silu", name="root"))
            else:
                producer = fm.F.math.vectorized_binary(x, x, binary_op="mul", name="producer")
                root = fm.F.ntt.vectorized_cast(producer, fm.vector_type("float32", (2, 4)), (1, ), name="root")
            self.function("main", (x, ), (root, producer) if exported else (root, ))

    module = Graph(dialect="high_level", stage="frozen_constants", entry="main").build()
    points, records, nodes = [], [], []
    for node in module.nodes:
        candidate = f"distribution.{node.id}.materialized"
        point = f"distribution.{node.id}"
        nodes.append(replace(node, metadata={**node.metadata, "distributed_candidate": candidate}))
        points.append(fm.SelectionPoint(point, "distribution", (fm.Candidate(candidate), ), candidate, node.id))
        records.append(fm.SelectionRecord(point, candidate, "default-policy", "test-layout/v1"))
    return fm.verify_module(
        replace(module, nodes=tuple(nodes), selection_points=tuple(points), selections=tuple(records)))


@pytest.mark.parametrize("mode", ["pre", "pre_repeated", "post"])
def test_fused_region_invalidates_only_its_original_selections(mode, tmp_path):
    original = _graph(mode)
    result = DataflowPass("Fuse", fusion_rules(), rewrite_constants=False).run(original)
    assert "producer" not in result.node_map
    assert {point.owner for point in result.selection_points} == {"x"}
    assert result.selections == (original.selections[0], )
    assert fm.load_module(fm.emit_module(result, tmp_path / "fused.py")) == result
    value = (torch.arange(64).reshape(2, 4, 8) / 8).bfloat16()
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(result, {"x": value}), evaluator.run(original, {"x": value}), rtol=0,
                               atol=0)


@pytest.mark.parametrize("mode", ["pre", "pre_repeated", "post"])
def test_exported_producer_keeps_its_selection_and_precision(mode):
    original = _graph(mode, exported=True)
    result = DataflowPass("Fuse", fusion_rules(), rewrite_constants=False).run(original)
    assert result == original
