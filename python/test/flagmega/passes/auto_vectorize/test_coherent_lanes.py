# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Projection packing is a region layout, not an independent byte width per op."""

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.selection import override_plan


def _module():

    class Graph(fm.Module):

        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (1, 64)))
            residual = self.input("residual", fm.tensor_type("float32", (1, 128)))
            w = self.weight("w", fm.tensor_type("bfloat16", (64, 128)), source="memory", key="w")
            scale = self.weight("scale", fm.tensor_type("bfloat16", (128, )), source="memory", key="scale")
            bias = fm.F.builtin.splat_const(fm.tensor_type("bfloat16", (128, )), 0.)
            projection = fm.F.math.matmul(x, w, output_data_type="float32", name="projection")
            value = fm.F.math.add(residual, projection, name="value")
            stats = fm.F.nn.norm_stats(value, axis=-1, use_mean=False, name="stats")
            output = fm.F.nn.norm_apply(value, stats, scale, bias, axis=-1, epsilon=1e-6, use_mean=False,
                                        output_dtype="bfloat16", name="output")
            self.function("main", (x, residual), (value, output))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def test_default_uses_producer_lanes_through_residual_and_norm(tmp_path):
    original = _module()
    compiler = Compiler()
    proposal = compiler.compile(original, stop_after="propose-vectorization").module
    assert proposal.selection_map["vectorization.value"].candidate_id == "vectorization.last_axis.lanes_2_4"
    # Both the default and agent alternatives survive Python edit/resume.
    proposal = fm.load_module(fm.emit_module(proposal, tmp_path / "proposal.py"))
    vectorized = compiler.compile(proposal, stop_after="apply-vectorization").module
    projection = vectorized.node_map["projection.vectorized.compute"]
    residual = vectorized.node_map["value.vectorized.compute"]
    assert residual.inputs[1] == projection.id
    assert residual.type == projection.type == fm.tensor_type(fm.vector_type("float32", (2, 4)), (1, 16))
    stats = vectorized.node_map["stats"]
    norm = vectorized.node_map["output.vectorized.compute"]
    assert stats.inputs[0] == norm.inputs[0] == residual.id
    assert not any(node.op == "tensors.pack" and node.inputs[0] in {projection.id, residual.id}
                   for node in vectorized.nodes)
    generator = torch.Generator().manual_seed(10)
    feeds = {
        "x": torch.randn(1, 64, generator=generator).bfloat16(), "residual": torch.randn(1, 128, generator=generator)
    }
    evaluator = TorchEvaluator(
        DictWeightResolver({
            "w": torch.randn(64, 128, generator=generator).bfloat16(), "scale":
            torch.randn(128, generator=generator).bfloat16()
        }))
    torch.testing.assert_close(evaluator.run(vectorized, feeds), evaluator.run(original, feeds), rtol=0, atol=0)


@pytest.mark.parametrize("candidate", ["vectorization.last_axis", "vectorization.scalar"])
def test_explicit_agent_choice_remains_authoritative(candidate):
    compiler = Compiler()
    proposal = compiler.compile(_module(), stop_after="propose-vectorization").module
    selected = compiler.run_stage(proposal, "apply-vectorization",
                                  plan=override_plan(proposal, (("vectorization.value", candidate), ))).module
    assert selected.selection_map["vectorization.value"].candidate_id == candidate


def test_tuple_call_result_and_later_consumer_share_producer_packet_layout():

    class Graph(fm.Module):

        def forward(self):
            narrow = fm.tensor_type("bfloat16", (1, 32))
            wide = fm.tensor_type("float32", (1, 32))
            matrix = fm.tensor_type("bfloat16", (32, 32))
            lhs, rhs, residual = (self.input(name, dtype)
                                  for name, dtype in (("lhs", narrow), ("rhs", matrix), ("residual", wide)))
            projected = fm.F.math.matmul(lhs, rhs, output_data_type="float32", name="projected")
            output = fm.F.math.add(projected, residual, name="sum")
            self.function("worker", (lhs, rhs, residual), (output, lhs), attrs={"reusable": True})
            x, w, r = (self.input(name, dtype) for name, dtype in (("x", narrow), ("w", matrix), ("r", wide)))
            first = fm.F.builtin.call(x, w, r, callee="worker", result_type=fm.TupleType((wide, narrow)))
            second = fm.F.builtin.call(x, w, fm.F.tensors.get_item(first, 0), callee="worker", result_type=fm.TupleType(
                (wide, narrow)))
            stats = fm.F.nn.norm_stats(fm.F.tensors.get_item(second, 0), axis=-1, use_mean=False, name="final_stats")
            self.function("main", (x, w, r), (stats, ))

    module = Graph(dialect="high_level", stage="imported", entry="main").build()
    proposal = Compiler().compile(module, stop_after="propose-vectorization").module
    assert proposal.selection_map["vectorization.final_stats"].candidate_id.endswith(".lanes_2_4")
