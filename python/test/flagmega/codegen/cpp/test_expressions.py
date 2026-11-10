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


def test_index_int64_widening_requires_a_typed_scalar_source():
    from triton.flagmega.codegen.cpp.templates import CppTemplateRegistry
    template = CppTemplateRegistry().environment.from_string("{{ 'layer_id.to(tl.int64) * 61440 // 2' | cppexpr }}")
    assert template.render(argument_types={'layer_id':{'type':'int','pointer':False}}) == '((static_cast<long long>(layer_id) * 61440LL) / 2LL)'
    for info in ({'type':'float','pointer':False}, {'type':'int','pointer':True}):
        with pytest.raises(CodegenError):
            template.render(argument_types={'layer_id':info})


def test_index_int64_widening_is_shared_by_call_argument_filter():
    from triton.flagmega.codegen.cpp.templates import CppTemplateRegistry
    template = CppTemplateRegistry().environment.from_string("{{ 'layer_id.to(tl.int64), pool.to(tl.pointer_type(tl.bfloat16))' | cppargs }}")
    assert template.render(argument_types={'layer_id':{'type':'int','pointer':False}}) == 'static_cast<long long>(layer_id), reinterpret_cast<ftt::bfloat16_t *>(pool)'


def test_untyped_or_numeric_float_casts_remain_rejected_in_index_expressions():
    with pytest.raises(CodegenError):
        translate('layer_id.to(tl.int64)')
    with pytest.raises(CodegenError):
        translate('layer_id.to(tl.float32)')


def test_typed_integer_span_is_exact_and_compiles_for_cuda(tmp_path):
    import ctypes
    import shutil
    import subprocess
    from triton.flagmega.codegen.cpp.templates import CppTemplateRegistry
    if shutil.which('c++') is None:
        pytest.skip('A host C++ compiler is required')
    template = CppTemplateRegistry().environment.from_string("{{ 'layer_id.to(tl.int64) * 61440 // 2' | cppexpr }}")
    expression = template.render(argument_types={'layer_id':{'type':'int','pointer':False}})
    source = tmp_path/'span.cpp'
    source.write_text(f'extern "C" long long span(int layer_id) {{return {expression};}}\n')
    library = tmp_path/'span.so'
    subprocess.run(['c++','-std=c++17','-shared','-fPIC',str(source),'-o',str(library)],check=True,capture_output=True,text=True)
    span = ctypes.CDLL(str(library)).span
    span.restype = ctypes.c_longlong
    for layer_id in (0,47,65536,2**31-1):
        assert span(layer_id) == layer_id*61440//2
    nvcc = shutil.which('nvcc')
    if nvcc:
        cuda = tmp_path/'span.cu'
        cuda.write_text(f'extern "C" __global__ void span_cuda(int layer_id,long long *output) {{*output = {expression};}}\n')
        subprocess.run([nvcc,'-std=c++20','-arch=sm_120','-c',str(cuda),'-o',str(tmp_path/'span.o')],check=True,capture_output=True,text=True)
