"""Frontend tests for C-interoperable custom types (structs).

These run the AST -> TTIR path only (via ``run_parser``); the ``tt-lower-structs``
pass that removes the custom types is covered by ``test/Triton/lower-structs.mlir``.
"""

import ctypes

import pytest
import triton
import triton.language as tl
from triton._filecheck import run_parser


@tl.struct_type
class TensorView(ctypes.Structure):
    _fields_ = [
        ("shape", ctypes.c_uint64 * 4),
        ("stride", ctypes.c_uint64 * 4),
        ("base", ctypes.c_void_p),
        ("ndim", ctypes.c_uint32),
    ]


@tl.struct_type
class Inner(ctypes.Structure):
    _fields_ = [("a", ctypes.c_int32), ("b", ctypes.c_double)]


@tl.struct_type
class Outer(ctypes.Structure):
    _fields_ = [("inner", Inner), ("n", ctypes.c_uint32)]


class _FakeBuffer:
    """Minimal stand-in for a device buffer: just needs data_ptr()."""

    def __init__(self, ptr=0x1000):
        self._ptr = ptr

    def data_ptr(self):
        return self._ptr


def _view(cls, ptr=0x1000):
    return tl.StructView(cls._triton_struct_type, _FakeBuffer(ptr))


def test_struct_layout_and_signature():
    st = TensorView._triton_struct_type
    assert st.size == 80
    assert st.align == 8
    assert [(n, o) for (n, _, o) in st.fields] == [("shape", 0), ("stride", 32), ("base", 64), ("ndim", 72)]
    # Round-trips through the self-contained signature string.
    from triton.language import str_to_ty
    ty = str_to_ty("*" + st.signature(), None)
    assert ty.element_ty == st


def test_read_scalar_field():
    @triton.jit
    def kernel(x_view):
        return x_view.ndim

    module = run_parser(kernel, args=(_view(TensorView),))
    text = module.str_nodebug()
    assert "tt.struct_gep" in text
    assert '"ndim"' in text
    # The struct pointer argument type is preserved through codegen.
    assert "!tt.struct<{shape:" in text


def test_write_scalar_field():
    @triton.jit
    def kernel(x_view):
        n = x_view.ndim
        x_view.ndim = n + 1
        return n

    text = run_parser(kernel, args=(_view(TensorView),)).str_nodebug()
    # One gep for the read and one for the write.
    assert text.count("tt.struct_gep") == 2
    assert "tt.store" in text


def test_nested_field():
    @triton.jit
    def kernel(o):
        return o.inner.a

    text = run_parser(kernel, args=(_view(Outer),)).str_nodebug()
    # A gep for the nested struct, then a gep into it.
    assert text.count("tt.struct_gep") == 2


def test_pointer_field():
    @triton.jit
    def kernel(x_view):
        return x_view.base

    text = run_parser(kernel, args=(_view(TensorView),)).str_nodebug()
    assert "tt.struct_gep" in text
    assert '"base"' in text


def test_unknown_field_raises():
    @triton.jit
    def kernel(x_view):
        return x_view.nope

    with pytest.raises(Exception):
        run_parser(kernel, args=(_view(TensorView),))
