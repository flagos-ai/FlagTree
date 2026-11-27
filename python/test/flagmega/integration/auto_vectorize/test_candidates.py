# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.selection import apply_plan, override_plan
from triton.flagmega.rules.ntt.vectorize.policy import NttVectorizationPolicy
from triton.flagmega.targets import NvidiaSm90Target


def _add_module(shape):
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("bfloat16", shape)
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    output = builder.call("math.add", [lhs, rhs], value_type, id="output")
    builder.function("main", [lhs, rhs], [output])
    return builder.build(entry="main")


def test_generic_ntt_policy_can_select_multi_axis_egraph_candidate():
    module = _add_module((16, 16))
    target = NvidiaSm90Target(
        vectorization_policy=NttVectorizationPolicy(lane_bytes=16, max_axes=2)
    )
    candidates = target.propose_vectorization(module)
    point = next(point for point in candidates.selection_points if point.id == "vectorization.output")
    assert {candidate.id for candidate in point.candidates}.issuperset({
        "vectorization.scalar", "vectorization.last_axis", "vectorization.axes_0_1",
    })
    plan = override_plan(candidates, ((point.id, "vectorization.axes_0_1"),))
    vectorized = target.apply_vectorization(apply_plan(candidates, plan))
    compute_type = vectorized.node_map["output.vectorized.compute"].type
    assert compute_type == fm.tensor_type(fm.vector_type("bfloat16", (8, 8)), (2, 2))

    lhs = torch.randn(16, 16, dtype=torch.bfloat16)
    rhs = torch.randn(16, 16, dtype=torch.bfloat16)
    output = TorchEvaluator(DictWeightResolver({})).run(vectorized, {"lhs": lhs, "rhs": rhs})[0]
    torch.testing.assert_close(output, lhs + rhs)


def test_symbolic_exact_division_can_vectorize_without_static_shape():
    n = fm.dim("n", minimum=1, maximum=64)
    module = _add_module((n * 8,))
    vectorized = Compiler().compile(module, stop_after="apply-vectorization").module
    assert vectorized.selection_map["vectorization.output"].candidate_id == "vectorization.last_axis"
    assert all(node.op != "tensors.pad" for node in vectorized.nodes)

    lhs = torch.randn(24, dtype=torch.bfloat16)
    rhs = torch.randn(24, dtype=torch.bfloat16)
    output = TorchEvaluator(DictWeightResolver({})).run(vectorized, {"lhs": lhs, "rhs": rhs})[0]
    torch.testing.assert_close(output, lhs + rhs)
