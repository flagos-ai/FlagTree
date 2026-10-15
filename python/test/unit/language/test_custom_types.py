"""Frontend tests for C-interoperable custom types (structs).

These run the AST -> TTIR path only (via ``run_parser``); the ``tt-lower-structs``
pass that removes the custom types is covered by ``test/Triton/lower-structs.mlir``.
"""

import ctypes
import os
import subprocess
import sys

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
    # float32 (not float64): Ascend's BiShengIR pipeline does not support
    # 64-bit float arithmetic.
    _fields_ = [("a", ctypes.c_int32), ("b", ctypes.c_float)]


@tl.struct_type
class Outer(ctypes.Structure):
    _fields_ = [("inner", Inner), ("n", ctypes.c_uint32)]


@tl.struct_type
class WithArray(ctypes.Structure):
    _fields_ = [("arr", Inner * 2), ("n", ctypes.c_uint32)]


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

    module = run_parser(kernel, args=(_view(TensorView), ))
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

    text = run_parser(kernel, args=(_view(TensorView), )).str_nodebug()
    # One gep for the read and one for the write.
    assert text.count("tt.struct_gep") == 2
    assert "tt.store" in text


def test_nested_field():

    @triton.jit
    def kernel(o):
        return o.inner.a

    text = run_parser(kernel, args=(_view(Outer), )).str_nodebug()
    # A gep for the nested struct, then a gep into it.
    assert text.count("tt.struct_gep") == 2


def test_pointer_field():

    @triton.jit
    def kernel(x_view):
        return x_view.base

    text = run_parser(kernel, args=(_view(TensorView), )).str_nodebug()
    assert "tt.struct_gep" in text
    assert '"base"' in text


def test_unknown_field_raises():

    @triton.jit
    def kernel(x_view):
        return x_view.nope

    with pytest.raises(Exception):
        run_parser(kernel, args=(_view(TensorView), ))


def test_array_element_read():

    @triton.jit
    def kernel(x_view):
        return x_view.shape[0]

    text = run_parser(kernel, args=(_view(TensorView), )).str_nodebug()
    # struct_gep for the "shape" field, then array_gep for the element.
    assert "tt.struct_gep" in text
    assert "tt.array_gep" in text


def test_array_element_dynamic_index():

    @triton.jit
    def kernel(x_view, i):
        return x_view.shape[i]

    text = run_parser(kernel, args=(_view(TensorView), 0)).str_nodebug()
    assert "tt.array_gep" in text


def test_array_element_write():

    @triton.jit
    def kernel(x_view):
        x_view.shape[0] = 7

    text = run_parser(kernel, args=(_view(TensorView), )).str_nodebug()
    assert "tt.array_gep" in text
    assert "tt.store" in text


def test_array_of_struct_element():

    @triton.jit
    def kernel(o, i):
        return o.arr[i].b

    text = run_parser(kernel, args=(_view(WithArray), 0)).str_nodebug()
    # struct_gep for "arr", array_gep for the element, struct_gep for "b".
    assert text.count("tt.struct_gep") == 2
    assert "tt.array_gep" in text


def test_array_of_struct_type_model():
    st = WithArray._triton_struct_type
    arr_ty = st.fields[0][1]
    assert isinstance(arr_ty.element_ty, type(Inner._triton_struct_type))
    assert arr_ty.count == 2
    from triton.language import str_to_ty
    assert str_to_ty("*" + st.signature(), None).element_ty == st


class _FakeDeviceBuffer(bytearray):
    """A writable byte buffer with a data_ptr(), mimicking a device tensor."""

    def data_ptr(self):
        return ctypes.addressof((ctypes.c_ubyte * len(self)).from_buffer(self))


def test_to_device_uploads_struct_bytes():
    tv = TensorView()
    tv.ndim = 7
    tv.base = 0xDEADBEEF
    st = TensorView._triton_struct_type
    buf = _FakeDeviceBuffer(st.size)
    view = tv.to_device(buffer=buf)
    assert isinstance(view, tl.StructView)
    assert view.nbytes == st.size
    raw = ctypes.string_at(ctypes.addressof(tv), ctypes.sizeof(TensorView))
    assert bytes(buf) == raw
    assert view.data_ptr() == buf.data_ptr()


def test_structview_signature():
    from triton.runtime.jit import mangle_type
    view = _view(TensorView)
    sig = mangle_type(view)
    assert sig.startswith("*struct<")
    assert "ndim:u32@72" in sig


# --- interpreter -----------------------------------------------------------------


@triton.jit
def _k_write_ndim(x_view):
    x_view.ndim = x_view.ndim + 1


@triton.jit
def _k_write_shape(x_view, i):
    x_view.shape[i] = x_view.shape[i] + 10


@triton.jit
def _k_nested_write(o):
    o.inner.a = o.inner.a + 3


@triton.jit
def _k_array_of_struct(o, i):
    o.arr[i].b = o.arr[i].b + 1.5


@triton.jit
def _k_copy_base(x_view):
    x_view.shape[3] = x_view.base


def _interp(fn):
    from triton.runtime.interpreter import InterpretedFunction
    return InterpretedFunction(fn)


# The Triton interpreter patches ``triton.language`` in place the first time an
# interpreted kernel runs, and never undoes it.  Running the interpreter here
# would therefore corrupt every compiled test that executes later in this
# process, so each case is executed in a child process instead.
_INTERPRETER_HELPER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "custom_types_interpreter_helper.py")


def _run_interpreter_case(case):
    proc = subprocess.run([sys.executable, _INTERPRETER_HELPER, case], capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode()


def test_interpreter_scalar_and_array_field():
    _run_interpreter_case("scalar_and_array_field")


def test_interpreter_nested_struct():
    _run_interpreter_case("nested_struct")


def test_interpreter_array_of_struct():
    _run_interpreter_case("array_of_struct")


def test_interpreter_pointer_field():
    _run_interpreter_case("pointer_field")


# --- real hardware (Ascend NPU) --------------------------------------------------


def _ascend_device():
    """Return the current Ascend device string, or None if unavailable."""
    try:
        import torch
        import torch_npu  # noqa: F401
        if torch.npu.is_available() and torch.npu.device_count() > 0:
            return f"npu:{torch.npu.current_device()}"
    except Exception:
        pass
    return None


requires_ascend = pytest.mark.skipif(_ascend_device() is None, reason="requires an Ascend NPU (torch_npu + CANN)")


@requires_ascend
def test_hardware_struct_read_write():
    """Launch a real kernel that reads and writes struct fields."""
    import torch
    device = _ascend_device()

    tv = TensorView()
    tv.ndim = 7
    tv.shape[0] = 3
    view = tv.to_device(device)

    @triton.jit
    def kernel(x_view):
        x_view.ndim = x_view.ndim + 1  # scalar field write
        x_view.shape[0] = x_view.shape[0] + 10  # array element write

    kernel[(1, )](view)
    torch.npu.synchronize()

    # Copy the struct bytes back and decode them with ctypes: the kernel must
    # have written the host/C layout exactly.
    host = bytearray(view.buffer.cpu().numpy().tobytes())
    result = TensorView.from_buffer(host)
    assert result.ndim == 8
    assert result.shape[0] == 13


@requires_ascend
def test_hardware_nested_and_array_of_struct():
    import torch
    device = _ascend_device()

    outer = Outer()
    outer.inner.a = 10
    view_o = outer.to_device(device)

    arr = WithArray()
    arr.arr[1].b = 3.5
    view_a = arr.to_device(device)

    @triton.jit
    def nested_kernel(o):
        o.inner.a = o.inner.a + 3

    @triton.jit
    def array_of_struct_kernel(o, i):
        o.arr[i].b = o.arr[i].b + 1.5

    nested_kernel[(1, )](view_o)
    array_of_struct_kernel[(1, )](view_a, 1)
    torch.npu.synchronize()

    res_o = Outer.from_buffer(bytearray(view_o.buffer.cpu().numpy().tobytes()))
    assert res_o.inner.a == 13
    res_a = WithArray.from_buffer(bytearray(view_a.buffer.cpu().numpy().tobytes()))
    assert abs(res_a.arr[1].b - 5.0) < 1e-9
