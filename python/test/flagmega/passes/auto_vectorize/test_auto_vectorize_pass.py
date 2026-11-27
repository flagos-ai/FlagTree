# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.auto_vectorize import AutoVectorizePass
from triton.flagmega.passes.auto_vectorize import _default_candidate
from triton.flagmega.passes.auto_distributed.policy import lower_vectorization_contracts
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.rules.ntt.vectorize import VectorizeCandidate
from triton.flagmega.selection import apply_plan, override_plan
from triton.flagmega.targets import NvidiaSm90Target


def _module():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="decomposed", entry="main")

        def forward(self):
            value_type = fm.tensor_type("bfloat16", (2, 16))
            lhs = self.input("lhs", value_type, id="lhs")
            rhs = self.input("rhs", value_type, id="rhs")
            root = fm.F.math.add(lhs, rhs, name="root")
            self.function("main", (lhs, rhs), (root,))

    return Graph().build()


def test_proposal_is_idempotent_and_records_every_target_executable_candidate():
    target = NvidiaSm90Target()
    proposed = AutoVectorizePass.propose(_module(), target)
    point = proposed.selection_points[0]
    assert point.id == "vectorization.root"
    assert point.owner == "root"
    assert [candidate.id for candidate in point.candidates] == [
        "vectorization.scalar", "vectorization.axes_0", "vectorization.last_axis",
    ]
    assert proposed.selection_map[point.id].candidate_id == "vectorization.last_axis"
    assert AutoVectorizePass.propose(proposed, target) == proposed


def test_scalar_selection_keeps_original_graph():
    target = NvidiaSm90Target()
    proposed = AutoVectorizePass.propose(_module(), target)
    selected = apply_plan(
        proposed,
        override_plan(proposed, (("vectorization.root", "vectorization.scalar"),)),
    )
    result = AutoVectorizePass.run(selected, target)
    assert result.node_map["root"].op == "math.add"
    assert all("vectorized" not in node.op for node in result.nodes)


def test_selected_candidate_is_extracted_as_multi_node_egraph_alternative():
    target = NvidiaSm90Target()
    proposed = AutoVectorizePass.propose(_module(), target)
    result = AutoVectorizePass.run(proposed, target)
    assert [node.op for node in result.nodes] == [
        "builtin.var", "builtin.var", "tensors.pack", "tensors.pack",
        "math.vectorized_binary", "tensors.unpack",
    ]
    assert result.function_map["main"].outputs == ("root",)
    assert result.selection_points[0].owner == "root"


def test_apply_requires_a_selection_record():
    target = NvidiaSm90Target()
    proposed = replace(AutoVectorizePass.propose(_module(), target), selections=())
    with pytest.raises(IRVerificationError, match="has no applied selection"):
        AutoVectorizePass.run(proposed, target)


def test_apply_rejects_candidate_that_is_no_longer_legal():
    target = NvidiaSm90Target()
    proposed = AutoVectorizePass.propose(_module(), target)
    vector_type = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (2, 2))
    nodes = tuple(replace(node, type=vector_type) for node in proposed.nodes)
    edited = replace(proposed, nodes=nodes)
    with pytest.raises(IRVerificationError, match="is no longer legal"):
        AutoVectorizePass.run(edited, target)


def test_apply_ignores_non_vectorization_selection_points():
    target = NvidiaSm90Target()
    proposed = AutoVectorizePass.propose(_module(), target)
    unrelated = fm.SelectionPoint(
        "packing.root", "packing", (fm.Candidate("logical"),), "logical", owner="root",
    )
    result = AutoVectorizePass.run(
        replace(proposed, selection_points=(*proposed.selection_points, unrelated)), target,
    )
    assert result.node_map["root"].op == "tensors.unpack"


def test_apply_rejects_rule_not_registered_by_current_target():
    proposed = AutoVectorizePass.propose(_module(), NvidiaSm90Target())

    class MissingBinaryTarget(NvidiaSm90Target):
        def register_auto_vectorize_rules(self, registry):
            return None

    with pytest.raises(IRVerificationError, match="did not register selected vectorization rule"):
        AutoVectorizePass.run(proposed, MissingBinaryTarget())


def test_default_candidate_falls_back_to_registration_order():
    candidate = VectorizeCandidate("custom", "Rule", (0,), (8,), {}, {})
    assert _default_candidate((candidate,)) == "custom"


def test_normalization_recovers_semantics_after_cast_boundary_propagation():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="decomposed", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (32,)), id="value")
            cast = fm.F.tensors.cast(value, fm.DType.FLOAT32, name="cast")
            root = fm.F.math.silu(cast, name="root")
            self.function("main", (value,), (root,))

    original = Graph().build()
    target = NvidiaSm90Target()
    vectorized = AutoVectorizePass.run(AutoVectorizePass.propose(original, target), target)
    assert any(node.op == "ntt.vectorized_cast" for node in vectorized.nodes)

    normalized = fm.verify_module(lower_vectorization_contracts(vectorized))
    assert [node.op for node in normalized.nodes] == [
        "builtin.var",
        "tensors.pack",
        "ntt.vectorized_cast",
        "math.vectorized_unary",
        "tensors.unpack",
    ]
    value = torch.randn(32, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(normalized, {"value": value})[0],
        evaluator.run(original, {"value": value})[0],
    )


def test_normalization_uses_egraph_canonical_operand_instead_of_stale_provenance():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="decomposed", entry="main")

        def forward(self):
            value_type = fm.tensor_type("bfloat16", (2, 16))
            parameter_type = fm.tensor_type("bfloat16", (16,))
            value = self.input("value", value_type, id="value")
            scale = self.input("scale", parameter_type, id="scale")
            first_zero = fm.F.builtin.splat_const(parameter_type, 0.0, name="first_zero")
            second_zero = fm.F.builtin.splat_const(parameter_type, 0.0, name="second_zero")
            # Both zero operands are equal e-nodes.  Extraction is allowed to
            # retain only one of them, while vectorization provenance still
            # records the source-level operand id.
            stats = fm.F.nn.norm_stats(value, axis=-1, use_mean=False, name="stats")
            first = fm.F.nn.norm_apply(
                value,
                stats,
                scale,
                first_zero,
                axis=-1,
                epsilon=1e-6,
                use_mean=False,
                name="first",
            )
            output = fm.F.nn.norm_apply(
                value,
                stats,
                scale,
                second_zero,
                axis=-1,
                epsilon=2e-6,
                use_mean=False,
                name="output",
            )
            self.function("main", (value, scale), (first, output))

    original = Graph().build()
    target = NvidiaSm90Target()
    vectorized = AutoVectorizePass.run(AutoVectorizePass.propose(original, target), target)
    assert "second_zero" not in vectorized.node_map
    assert vectorized.node_map["output"].metadata["vectorization_inputs"][-1] == "second_zero"

    normalized = fm.verify_module(lower_vectorization_contracts(vectorized))
    compute = normalized.node_map["output.vectorized.compute"]
    canonical_bias_pack = normalized.node_map[compute.inputs[-1]]
    assert canonical_bias_pack.op == "tensors.pack"
    assert canonical_bias_pack.inputs == ("first_zero",)
