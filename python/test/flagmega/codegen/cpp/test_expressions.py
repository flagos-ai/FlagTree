# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The renderer-expression-to-C++ translation, which both backends depend on.

These are unit tests rather than codegen tests because the translation is the
one place where the two backends' shared index algebra can silently diverge: a
Python `//` emitted into C++ computes a different value for the same bytes.
"""

import pytest

from triton.flagmega.codegen.cpp.expressions import cpp_scalar_type, translate
from triton.flagmega.errors import CodegenError


@pytest.mark.parametrize("source,expected", [
    # Integer literals carry a width suffix so a 64-bit stride product does
    # not overflow an int on the way to a pointer.
    ("0", "0LL"),
    ("offsets", "offsets"),
    ("offsets + 1", "(offsets + 1LL)"),
    # Python floor division maps to C++ `/` only because every operand is a
    # non-negative index; see the guard tests below.
    ("offsets // 17", "(offsets / 17LL)"),
    ("offsets % 17", "(offsets % 17LL)"),
    # `&`/`|` are Triton's elementwise boolean operators, not bitwise ones.
    ("(a < 3) & (b < 4)", "((a < 3LL) && (b < 4LL))"),
    ("(a < 3) | (b < 4)", "((a < 3LL) || (b < 4LL))"),
    ("tl.where(a < 3, a, 0)", "((a < 3LL) ? a : 0LL)"),
    ("tl.maximum(a, b)", "ftt::max(a, b)"),
])
def test_translates_renderer_index_algebra(source, expected):
    assert translate(source) == expected


def test_translates_pointer_retype_to_cast():
    assert translate("pool.to(tl.pointer_type(tl.bfloat16))") == (
        "reinterpret_cast<ftt::bfloat16_t *>(pool)"
    )


def test_drops_broadcast_subscripts():
    """A 2-D tile address is two scalars in C++, so the broadcast is identity."""

    assert translate("row[:, None] * 8 + col[None, :]") == "((row * 8LL) + col)"


def test_rejects_element_selecting_subscript():
    """Erasing a real subscript would read the wrong address, so it must raise."""

    with pytest.raises(CodegenError, match="not a broadcast"):
        translate("table[index]")


def test_rejects_subtraction_under_floor_division():
    """Floor and truncating division disagree for a negative dividend."""

    with pytest.raises(CodegenError, match="subtraction under floor division"):
        translate("(a - b) // 4")


def test_rejects_unknown_triton_intrinsic():
    with pytest.raises(CodegenError, match="tl.histogram"):
        translate("tl.histogram(values)")


def test_tl_full_collapses_to_its_value():
    """`tl.full` keeps a literal dynamic across a Triton function boundary.

    A C++ parameter is already dynamic, so only the value survives.
    """

    assert translate("tl.full((), 7, tl.int32)") == "7LL"


def test_scalar_dtype_spellings():
    assert cpp_scalar_type("bfloat16") == "ftt::bfloat16_t"
    assert cpp_scalar_type("int32") == "int"
    with pytest.raises(CodegenError, match="float8"):
        cpp_scalar_type("float8_e4m3fn")
