# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.rules.ntt.vectorize import VectorizeBinary


@pytest.mark.parametrize("op", ["math.add", "math.mul"])
@pytest.mark.parametrize("dtype,lane", [("bfloat16", 8), ("float32", 4)])
def test_binary_rule_enumerates_single_and_double_axis_candidates(make_op_module, op, dtype, lane):
    module = make_op_module(op, (fm.tensor_type(dtype, (5, 10)),) * 2)
    candidates = VectorizeBinary().candidates(module.node_map["root"], module)
    by_id = {candidate.id: candidate for candidate in candidates}
    assert set(by_id) == {"vectorization.axes_0", "vectorization.last_axis", "vectorization.axes_0_1"}
    assert by_id["vectorization.last_axis"].lanes == (lane,)
    assert by_id["vectorization.axes_0_1"].facts["padding"] == [(-5) % lane, (-10) % lane]


def test_binary_rule_honors_max_axes(make_op_module):
    value_type = fm.tensor_type("bfloat16", (8, 8, 8))
    module = make_op_module("math.add", (value_type, value_type))
    candidates = VectorizeBinary(max_axes=1).candidates(module.node_map["root"], module)
    assert {candidate.axes for candidate in candidates} == {(0,), (1,), (2,)}


@pytest.mark.parametrize("candidate_id", ["vectorization.last_axis", "vectorization.axes_0_1"])
@pytest.mark.parametrize("op", ["math.add", "math.mul"])
def test_binary_rule_rewrite_is_structural_and_numerically_equivalent(
    make_op_module, rewrite_candidate, op, candidate_id,
):
    value_type = fm.tensor_type("bfloat16", (8, 16))
    module = make_op_module(op, (value_type, value_type))
    candidate, result, rewritten = rewrite_candidate(module, VectorizeBinary(), candidate_id)

    assert result.replacement.op == "tensors.unpack"
    assert [node.op for node in result.prefix_nodes] == [
        "tensors.pack", "tensors.pack", "math.vectorized_binary",
    ]
    compute = result.prefix_nodes[-1]
    assert compute.attrs["binary_op"] == op.removeprefix("math.")
    assert compute.metadata["vectorization_root"] == "root"
    assert result.replacement.metadata["vectorization_candidate"] == candidate.id

    lhs = torch.randn(8, 16, dtype=torch.bfloat16)
    rhs = torch.randn(8, 16, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, {"arg0": lhs, "arg1": rhs})[0],
        evaluator.run(module, {"arg0": lhs, "arg1": rhs})[0],
    )


def test_binary_rule_rewrite_materializes_pad_and_slice(make_op_module, rewrite_candidate):
    value_type = fm.tensor_type("bfloat16", (3, 10))
    module = make_op_module("math.add", (value_type, value_type))
    candidate, result, rewritten = rewrite_candidate(module, VectorizeBinary(), "vectorization.last_axis")

    assert candidate.facts["padding"] == [0, 6]
    assert [node.op for node in result.prefix_nodes] == [
        "tensors.pad", "tensors.pack", "tensors.pad", "tensors.pack",
        "math.vectorized_binary", "tensors.unpack",
    ]
    assert result.replacement.op == "tensors.slice_to_shape"
    assert result.replacement.attrs["shape"] == (3, 10)

    lhs = torch.randn(3, 10, dtype=torch.bfloat16)
    rhs = torch.randn(3, 10, dtype=torch.bfloat16)
    output = TorchEvaluator(DictWeightResolver({})).run(rewritten, {"arg0": lhs, "arg1": rhs})[0]
    torch.testing.assert_close(output, lhs + rhs)


def test_binary_rule_does_not_match_other_operations(make_op_module):
    module = make_op_module("math.silu", (fm.tensor_type("bfloat16", (16,)),))
    assert VectorizeBinary().candidates(module.node_map["root"], module) == ()
