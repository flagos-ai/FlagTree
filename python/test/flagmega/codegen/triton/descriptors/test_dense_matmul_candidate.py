# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.targets import NvidiaSm90Target


def _proposed_tir(*, reduction: int = 128, output: int = 128) -> fm.IRModule:
    # Host descriptors read logical affine weight tiles. Already-packed
    # VectorizedMatMul operands exercise a different candidate path.
    builder = fm.IRBuilder(dialect="high_level", stage="distributed")
    lhs = builder.var("lhs", fm.tensor_type("bfloat16", (1, reduction)), id="lhs")
    rhs = builder.var("rhs", fm.tensor_type("bfloat16", (reduction, output)), id="rhs")
    result = builder.call(
        "math.matmul",
        (lhs, rhs),
        fm.tensor_type("bfloat16", (1, output)),
        id="result",
        attrs={"transpose_a": False, "transpose_b": False},
        metadata={
            "selected_vectorization": "vectorization.matmul.n", "selected_vector_axes": (1, ), "selected_vector_lanes":
            (8, )
        },
    )
    builder.function("main", (lhs, rhs), (result, ))
    return NvidiaSm90Target().propose_tir(builder.build(entry="main"))


def test_descriptor_variant_is_real_but_not_the_unmeasured_default():
    proposed = _proposed_tir()
    point = next(value for value in proposed.selection_points if value.id == "tir.result")

    assert point.default_candidate == "tir.dense_matmul.gemv"
    descriptor = next(value for value in point.candidates if value.id == "tir.dense_matmul.tensor_descriptor_gemv")
    assert descriptor.facts["requires"] == ("tma", )
    assert descriptor.facts["host_tensor_descriptor"] is True
    assert descriptor.parameters["family"] == "dense_matmul"
    assert descriptor.parameters["variant"] == "tensor_descriptor_gemv"


def test_descriptor_variant_requires_complete_local_tiles():
    proposed = _proposed_tir(reduction=192, output=120)
    point = next(value for value in proposed.selection_points if value.id == "tir.result")

    assert "tir.dense_matmul.tensor_descriptor_gemv" not in {value.id for value in point.candidates}
