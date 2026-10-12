#!/usr/bin/env python3
"""Demo: pass a C struct to a Triton kernel by reference (Ascend NPU).

The host and the kernel share a single C struct definition:

    struct TensorView {
        float  *base;      // pointer to the tensor data
        int32_t size;      // number of elements
        int32_t shape[4];  // shape (a fixed-size array field)
        int32_t counter;   // updated by the kernel
    };

The host fills the struct in, uploads it with ``tl.to_device``, and passes it
to the kernel.  The kernel reads its fields (including the ``base`` pointer,
which it dereferences) and writes back to ``counter`` and ``shape[0]``.  The
host then copies the struct back and reads the updated fields.

Because both sides use the same C layout, the bytes line up exactly.

Run on a machine with an Ascend NPU + torch_npu + CANN:
    python demo_custom_types.py

See docs/custom-types/RFC.md.
"""

import ctypes

import torch
import torch_npu

import triton
import triton.language as tl

DEVICE = "npu"


# ---------------------------------------------------------------------------
# The C struct.  One definition, used by both the host and the kernel.
# ---------------------------------------------------------------------------
@tl.struct_type
class TensorView(ctypes.Structure):
    _fields_ = [
        ("base", ctypes.POINTER(ctypes.c_float)),  # float* into the data
        ("size", ctypes.c_int32),  # number of elements
        ("shape", ctypes.c_int32 * 4),  # fixed-size array field
        ("counter", ctypes.c_int32),  # a field the kernel updates
    ]


# ---------------------------------------------------------------------------
# The kernel receives a pointer to the struct and uses its fields.
# ---------------------------------------------------------------------------
@triton.jit
def double_kernel(view, out_ptr, BLOCK: tl.constexpr):
    n = view.size  # scalar field read
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(view.base + offs, mask=mask)  # pointer field, dereferenced
    tl.store(out_ptr + offs, x * 2.0, mask=mask)
    view.counter = view.counter + 1  # scalar field write
    view.shape[0] = n + 100  # array element write


def main():
    n = 8
    x = torch.arange(n, dtype=torch.float32, device=DEVICE)
    out = torch.zeros(n, dtype=torch.float32, device=DEVICE)

    # 1. Host: fill in the struct.
    view = TensorView()
    view.base = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_float))
    view.size = n
    view.shape[0] = n
    view.counter = 0

    # 2. Upload the struct and launch the kernel, passing it by reference.
    view_dev = tl.to_device(view, device=DEVICE)
    double_kernel[(1,)](view_dev, out, BLOCK=n)
    torch.npu.synchronize()

    # 3. Copy the struct back and decode it with the same ctypes type.
    view_back = TensorView.from_buffer(bytearray(view_dev.buffer.cpu().numpy().tobytes()))

    print("input       :", x.cpu().tolist())
    print("output (x2) :", out.cpu().tolist())
    print("counter     :", view_back.counter, "  (kernel incremented it)")
    print("shape[0]    :", view_back.shape[0], "  (kernel wrote size + 100)")

    assert out.cpu().tolist() == [2.0 * v for v in x.cpu().tolist()]
    assert view_back.counter == 1
    assert view_back.shape[0] == n + 100
    print("OK")


if __name__ == "__main__":
    main()
