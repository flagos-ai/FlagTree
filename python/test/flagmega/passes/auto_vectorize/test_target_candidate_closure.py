# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.selection import override_plan


def _elementwise_module(op: str) -> fm.IRModule:
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("bfloat16", (3, 17))
    lhs = builder.var("lhs", value_type, id="lhs")
    inputs = (lhs,)
    if op in {"math.add", "math.mul"}:
        rhs = builder.var("rhs", value_type, id="rhs")
        inputs = (lhs, rhs)
    output = builder.call(op, inputs, value_type, id="output")
    builder.function("main", inputs, (output,))
    return builder.build(entry="main")


def _matmul_module() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    lhs_type = fm.tensor_type("bfloat16", (1, 128))
    rhs_type = fm.tensor_type("bfloat16", (128, 128))
    output_type = fm.tensor_type("bfloat16", (1, 128))
    lhs = builder.var("lhs", lhs_type, id="lhs")
    rhs = builder.var("rhs", rhs_type, id="rhs")
    output = builder.call(
        "math.matmul",
        (lhs, rhs),
        output_type,
        id="output",
        attrs={"transpose_a": False, "transpose_b": False},
    )
    builder.function("main", (lhs, rhs), (output,))
    return builder.build(entry="main")


@pytest.mark.parametrize("op", ["math.add", "math.mul", "math.silu"])
def test_ntt_target_publishes_rank_parameterized_elementwise_layouts(op):
    proposed = Compiler().compile(
        _elementwise_module(op), stop_after="propose-vectorization"
    ).module
    point = next(value for value in proposed.selection_points if value.id == "vectorization.output")
    assert {candidate.id for candidate in point.candidates} == {
        "vectorization.scalar",
        "vectorization.axes_0",
        "vectorization.last_axis",
    }


def test_current_triton_target_matches_nncase_rank_one_matmul_registration():
    proposed = Compiler().compile(
        _matmul_module(), stop_after="propose-vectorization"
    ).module
    point = next(value for value in proposed.selection_points if value.id == "vectorization.output")
    assert {candidate.id for candidate in point.candidates} == {
        "vectorization.scalar",
        "vectorization.matmul.n",
    }


@pytest.mark.parametrize("candidate_id", ["vectorization.scalar", "vectorization.matmul.n"])
def test_every_published_matmul_layout_reaches_executable_tir(candidate_id):
    compiler = Compiler()
    proposed = compiler.compile(
        _matmul_module(), stop_after="propose-vectorization"
    ).module
    selected = override_plan(
        proposed, (("vectorization.output", candidate_id),)
    )
    applied = compiler.run_stage(
        proposed, "apply-vectorization", plan=selected
    ).module
    lowered = compiler.compile(applied).module
    dispatch = fm.kernel_dispatch_for_call(lowered, lowered.node_map["output"])
    assert dispatch is not None
    assert dispatch.parameters["vector_schedule"]["contract"]["kind"] == (
        "scalar" if candidate_id == "vectorization.scalar" else "output_axis"
    )


def test_semantic_rules_remain_target_independent_and_can_enumerate_other_axes():
    from triton.flagmega.rules.ntt.vectorize import VectorizeBinary, VectorizeMatMul

    binary = _elementwise_module("math.add")
    matmul = _matmul_module()
    assert {candidate.id for candidate in VectorizeBinary().candidates(
        binary.node_map["output"], binary
    )} == {
        "vectorization.axes_0",
        "vectorization.last_axis",
        "vectorization.axes_0_1",
    }
    assert {candidate.parameters["kind"] for candidate in VectorizeMatMul().candidates(
        matmul.node_map["output"], matmul
    )} == {"n", "mn", "mkn"}
