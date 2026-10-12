"""Real-hardware tests for C-interoperable custom struct types on Ascend NPU.

A ``ctypes.Structure`` declared with ``@tl.struct_type`` is uploaded to device
memory with ``tl.to_device`` and passed to the kernel by reference; the kernel
reads and writes its fields.  The struct is copied back and decoded with ctypes
to confirm the bytes match the host/C layout.

See ``docs/custom-types/RFC.md``.
"""

import ctypes

import torch
import torch_npu

import triton
import triton.language as tl


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


DEVICE = "npu"


def _to_host(view):
    """Copy a StructView's device buffer back to a writable host buffer."""
    return bytearray(view.buffer.cpu().numpy().tobytes())


def test_struct_scalar_and_array_field():
    """Read and write a scalar field and an array element through the pointer."""
    tv = TensorView()
    tv.ndim = 7
    tv.shape[0] = 3
    view = tl.to_device(tv, device=DEVICE)

    @triton.jit
    def kernel(x_view):
        x_view.ndim = x_view.ndim + 1  # scalar field write
        x_view.shape[0] = x_view.shape[0] + 10  # array element write

    kernel[(1,)](view)
    torch.npu.synchronize()

    result = TensorView.from_buffer(_to_host(view))
    assert result.ndim == 8
    assert result.shape[0] == 13


def test_struct_nested_field():
    outer = Outer()
    outer.inner.a = 10
    view = tl.to_device(outer, device=DEVICE)

    @triton.jit
    def kernel(o):
        o.inner.a = o.inner.a + 3

    kernel[(1,)](view)
    torch.npu.synchronize()

    assert Outer.from_buffer(_to_host(view)).inner.a == 13


def test_struct_array_of_struct():
    arr = WithArray()
    arr.arr[1].b = 3.5
    view = tl.to_device(arr, device=DEVICE)

    @triton.jit
    def kernel(o, i):
        o.arr[i].b = o.arr[i].b + 1.5

    kernel[(1,)](view, 1)
    torch.npu.synchronize()

    result = WithArray.from_buffer(_to_host(view))
    assert abs(result.arr[1].b - 5.0) < 1e-9


def test_struct_pointer_field():
    """A pointer field (base) is read back and compared to the host address."""
    data = torch.arange(8, dtype=torch.int32, device=DEVICE)
    tv = TensorView()
    tv.base = data.data_ptr()
    view = tl.to_device(tv, device=DEVICE)

    @triton.jit
    def kernel(x_view):
        # Copy the raw pointer value into a uint64 field for comparison.
        x_view.shape[3] = x_view.base.to(tl.uint64)

    kernel[(1,)](view)
    torch.npu.synchronize()

    result = TensorView.from_buffer(_to_host(view))
    assert result.shape[3] == data.data_ptr()
