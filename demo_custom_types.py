#!/usr/bin/env python3
"""FlagTree demo: C-interoperable custom struct types.

A `TensorView` C struct (shape, stride, base, ndim) is defined once with
`@tl.struct_type` and shared between host and kernel.  The struct is uploaded
to the device, passed to a Triton kernel **by reference**, and the kernel
reads/writes its scalar, array and pointer fields.

    struct TensorView {            # one definition, host <-> kernel
        uint64 shape[4];
        uint64 stride[4];
        void  *base;
        uint32 ndim;
    };

Runs on an Ascend NPU when `torch_npu` is available, otherwise falls back to
the CPU Triton interpreter.

See docs/custom-types/RFC.md.
"""

import ctypes

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# 1. The C struct: one definition, shared between host and kernel.
# ---------------------------------------------------------------------------
@tl.struct_type
class TensorView(ctypes.Structure):
    _fields_ = [
        ("shape", ctypes.c_uint64 * 4),  # fixed-size array field
        ("stride", ctypes.c_uint64 * 4),
        ("base", ctypes.c_void_p),  # pointer field
        ("ndim", ctypes.c_uint32),  # scalar field
    ]


# ---------------------------------------------------------------------------
# 2. A kernel that uses the struct by reference.
# ---------------------------------------------------------------------------
@triton.jit
def process_kernel(x_view, out_ptr):
    ndim = x_view.ndim  # scalar field read
    n0 = x_view.shape[0]  # array element read
    # pointer field: reinterpret `base` (a void*) as a float32*
    base = tl.cast(tl.cast(x_view.base, tl.uint64), tl.pointer_type(tl.float32))
    offs = tl.arange(0, 8)
    v = tl.load(base + offs)
    tl.store(out_ptr + offs, v + ndim)  # use the metadata
    x_view.ndim = ndim + 1  # scalar field write
    x_view.stride[0] = n0 + 100  # array element write


# ---------------------------------------------------------------------------
# 3. Host side.
# ---------------------------------------------------------------------------
def print_overview():
    st = TensorView._triton_struct_type
    print("=" * 70)
    print("C-interoperable custom struct types (ctypes.Structure)")
    print("=" * 70)
    print("\nstruct TensorView -- one definition, shared host <-> kernel:")
    for name, ty, off in st.fields:
        print(f"    {name:<7} @{off:<3} : {ty}")
    print(f"    sizeof={st.size}  align={st.align}")
    print("\nsignature parsed by str_to_ty (kernel arg type):")
    print("    *" + st.signature())
    print("\nkernel field access lowers to IR ops:")
    print("    x_view.ndim      -> tt.struct_gep + load/store  (u32)")
    print("    x_view.shape[0]  -> tt.struct_gep + tt.array_gep + load/store  (u64)")
    print("    x_view.base      -> tt.struct_gep + i64 round-trip  (pointer)")
    print()


def _npu_available():
    try:
        import torch
        import torch_npu  # noqa: F401

        return torch.npu.is_available() and torch.npu.device_count() > 0
    except Exception:
        return False


def _report(ndim, stride0, out, expected):
    print(f"    ndim     : {ndim}  (expected 4)")
    print(f"    stride[0]: {stride0}  (expected 107)")
    print(f"    out      : {[round(float(x), 1) for x in out]}")
    print(f"    expected : {[round(float(x), 1) for x in expected]}")
    assert ndim == 4 and stride0 == 107
    assert all(abs(float(a) - float(b)) < 1e-5 for a, b in zip(out, expected))
    print("    OK")


def run_npu():
    import torch

    print("[exec] Ascend NPU")
    device = "npu"
    data = torch.arange(8, dtype=torch.float32, device=device)
    out = torch.zeros(8, dtype=torch.float32, device=device)

    tv = TensorView()
    tv.ndim = 3
    tv.shape[0] = 7
    tv.base = data.data_ptr()

    view = tl.to_device(tv, device=device)  # upload the struct bytes
    process_kernel[(1,)](view, out)  # pass by reference
    torch.npu.synchronize()

    # Copy the struct back and decode it with the same ctypes type.
    result = TensorView.from_buffer(bytearray(view.buffer.cpu().numpy().tobytes()))
    _report(result.ndim, result.stride[0], out.cpu().tolist(), (data.cpu() + 3.0).tolist())


class _HostBuffer(bytearray):
    """A writable host buffer with data_ptr() (mimics a device tensor)."""

    dtype = tl.float32

    def data_ptr(self):
        return ctypes.addressof((ctypes.c_ubyte * len(self)).from_buffer(self))


def run_interpreter():
    import numpy as np

    from triton.runtime.interpreter import InterpretedFunction

    print("[exec] CPU interpreter (no NPU found)")
    data = np.arange(8, dtype=np.float32)

    struct_buf = _HostBuffer(TensorView._triton_struct_type.size)
    tv = TensorView.from_buffer(struct_buf)
    tv.ndim = 3
    tv.shape[0] = 7
    tv.base = data.ctypes.data
    view = tl.StructView(TensorView._triton_struct_type, struct_buf)

    out_buf = _HostBuffer(8 * 4)
    out = np.frombuffer(out_buf, dtype=np.float32)

    InterpretedFunction(process_kernel.fn).run(view, out_buf, grid=(1,))
    _report(tv.ndim, tv.stride[0], out.tolist(), (data + 3.0).tolist())


def main():
    print_overview()
    if _npu_available():
        run_npu()
    else:
        run_interpreter()
    print("\nDone.")


if __name__ == "__main__":
    main()
