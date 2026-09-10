# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.rules.ntt.vectorize import VectorizeMatMul


def _matmul_module(make_op_module, lhs_shape=(8, 8), rhs_shape=(8, 8), *, dtype="bfloat16", **attrs):
    return make_op_module(
        "math.matmul",
        (fm.tensor_type(dtype, lhs_shape), fm.tensor_type(dtype, rhs_shape)),
        attrs=attrs,
    )


def test_matmul_rule_enumerates_nncase_rank_two_coupled_candidates(make_op_module):
    module = _matmul_module(make_op_module, (5, 7), (7, 10))
    candidates = VectorizeMatMul().candidates(module.node_map["root"], module)
    assert [candidate.id for candidate in candidates] == [
        "vectorization.matmul.n", "vectorization.matmul.mn",
        "vectorization.matmul.mkn",
    ]
    by_id = {candidate.id: candidate for candidate in candidates}
    assert by_id["vectorization.matmul.mn"].facts == {
        "lhs_padding": [3, 0], "rhs_padding": [0, 6],
        "output_padding": [3, 6], "egraph_equivalent": True,
    }


@pytest.mark.parametrize(
    "attrs,expected",
    [
        ({}, {"n": ([], [1])}),
        (
            {"transpose_a": True, "transpose_b": True},
            {"n": ([], [0])},
        ),
    ],
)
def test_matmul_rule_maps_axes_through_transpose(make_op_module, attrs, expected):
    module = _matmul_module(make_op_module, **attrs)
    by_kind = {
        candidate.parameters["kind"]: candidate
        for candidate in VectorizeMatMul(max_axes=1).candidates(module.node_map["root"], module)
    }
    for kind, (lhs_axes, rhs_axes) in expected.items():
        assert by_kind[kind].parameters["lhs_axes"] == lhs_axes
        assert by_kind[kind].parameters["rhs_axes"] == rhs_axes


@pytest.mark.parametrize("kind", ["n", "mn", "mkn"])
@pytest.mark.parametrize("output_data_type", [None, "float32"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_each_matmul_candidate_rewrites_and_evaluates_equivalently(
    make_op_module, rewrite_candidate, kind, output_data_type, dtype,
):
    module = _matmul_module(make_op_module, dtype=dtype, output_data_type=output_data_type)
    _, result, rewritten = rewrite_candidate(module, VectorizeMatMul(), f"vectorization.matmul.{kind}")
    assert any(node.op == "math.vectorized_matmul" for node in (*result.prefix_nodes, result.replacement))
    compute = next(node for node in rewritten.nodes if node.op == "math.vectorized_matmul")
    assert compute.attrs.get("output_data_type") == output_data_type

    if output_data_type == "float32":
        assert compute.type.dtype.lanes == ((2, 4) if kind == "n" else (8, 2, 4))
    lhs = torch.randn(8, 8, dtype=getattr(torch, dtype))
    rhs = torch.randn(8, 8, dtype=getattr(torch, dtype))
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, {"arg0": lhs, "arg1": rhs})[0],
        evaluator.run(module, {"arg0": lhs, "arg1": rhs})[0],
    )


def test_matmul_n_candidate_pads_rhs_and_slices_output(make_op_module, rewrite_candidate):
    module = _matmul_module(make_op_module, (5, 7), (7, 10))
    candidate, result, rewritten = rewrite_candidate(module, VectorizeMatMul(), "vectorization.matmul.n")
    assert candidate.facts["rhs_padding"] == [0, 6]
    assert result.replacement.op == "tensors.slice_to_shape"
    assert [node.op for node in result.prefix_nodes] == [
        "tensors.pad", "tensors.pack", "math.vectorized_matmul", "tensors.unpack",
    ]

    lhs = torch.randn(5, 7, dtype=torch.bfloat16)
    rhs = torch.randn(7, 10, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, {"arg0": lhs, "arg1": rhs})[0],
        evaluator.run(module, {"arg0": lhs, "arg1": rhs})[0],
    )


def test_matmul_rule_rejects_unsupported_dtype_lane_and_effect(make_op_module):
    int32 = make_op_module(
        "math.matmul", (fm.tensor_type("int32", (8, 8)), fm.tensor_type("int32", (8, 8))),
    )
    assert VectorizeMatMul().candidates(int32.node_map["root"], int32) == ()

    module = _matmul_module(make_op_module)
    assert VectorizeMatMul(lane_bytes=2).candidates(module.node_map["root"], module) == ()
    effectful = replace(module.node_map["root"], effect=fm.effect("write", "state"))
    assert VectorizeMatMul().candidates(effectful, replace(module, nodes=(*module.nodes[:-1], effectful))) == ()


def test_matmul_rule_filters_dynamic_axes_and_static_padding_with_dynamic_shape(make_op_module):
    n = fm.dim("n", minimum=1, maximum=64)
    exact = _matmul_module(make_op_module, (n * 8, 8), (8, 8))
    assert {candidate.parameters["kind"] for candidate in VectorizeMatMul().candidates(
        exact.node_map["root"], exact,
    )} == {"n", "mn", "mkn"}

    unproven = _matmul_module(make_op_module, (n, 7), (7, 10))
    assert {candidate.parameters["kind"] for candidate in VectorizeMatMul().candidates(
        unproven.node_map["root"], unproven,
    )} == {"n"}
