# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.printer import il_source


def _identity_module(value_type: fm.IRType):
    builder = fm.IRBuilder(dialect="high_level", stage="unit")
    value = builder.var("value", value_type, id="value")
    builder.function("main", [value], [value])
    return builder.build(entry="main")


@pytest.mark.parametrize(
    ("dtype", "display"),
    (
        ("bool", "bool"),
        ("int32", "i32"),
        ("int64", "i64"),
        ("bfloat16", "bf16"),
        ("float32", "f32"),
        ("float8_e4m3fn", "F8_E4M3"),
    ),
)
def test_scalar_dtype_names_match_nncase_il(dtype: str, display: str):
    source = il_source(_identity_module(fm.tensor_type(dtype, [1])))

    assert f"%value: {display}[1]" in source
    assert f"// ({display}[1]) -> ({display}[1])" in source
    assert "// (%value:" not in source


def test_tensor_type_uses_nncase_dtype_shape_and_vector_spelling():
    value_type = fm.tensor_type(fm.vector_type("float8_e4m3fn", (2, 16)), [10240, 160])

    source = il_source(_identity_module(value_type))

    assert "%value: F8_E4M3<2,16>[10240,160]" in source
    assert "float8_e4m3fn" not in source
    assert "VectorType(" not in source


def test_symbolic_shape_and_non_default_layout_are_compact_but_complete():
    batch = fm.dim("batch", minimum=1, maximum=64)
    value_type = fm.tensor_type(
        "bfloat16",
        [batch, 5120],
        layout=fm.TensorLayout(
            order=(1, 0),
            strides=(None, 1),
            vector_lanes=(8,),
            tag="packed",
        ),
    )

    source = il_source(_identity_module(value_type))

    assert (
        "%value: bf16[batch,5120]"
        "{layout=packed, order=[1,0], strides=[?,1], vector_lanes=[8]}"
    ) in source
    assert "DimVar(" not in source
    assert "TensorLayout(" not in source
