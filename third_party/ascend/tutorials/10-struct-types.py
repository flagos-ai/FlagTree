# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""
User-Defined Struct Types
=========================

In this tutorial, you will pass a C struct to a Triton kernel by reference.

A ``ctypes.Structure`` decorated with ``tl.struct_type`` is uploaded to device
memory with its ``to_device`` method and handed to the kernel as a pointer.  The
host and the kernel share the *same* C definition, so the bytes line up exactly:
no manual packing, no offset arithmetic on the host side.

This is handy whenever a kernel needs a small bundle of related arguments -- a
data pointer together with its metadata, a descriptor, a workspace, a set of
counters -- and you would rather pass one handle than a long argument list.  It
also lets the kernel update that state in place, so the host can read it back
after the launch.

In doing so, you will learn about:

* Defining a ``@tl.struct_type`` from a ``ctypes.Structure``.

* Giving it an ordinary ``__init__`` so the host can build one from a tensor.

* Reading scalar fields and fixed-size array fields.

* Dereferencing a pointer field to load a whole block of data.

* Writing fields back, and reading the updated struct from the host.
"""

# %%
# The struct
# ----------
#
# The kernel receives a device pointer to this struct.  ``ctypes`` describes the
# layout once; ``@tl.struct_type`` teaches Triton the same layout.  The
# ``__init__`` is ordinary Python that fills the fields in, so the host can
# build one straight from a tensor instead of poking at raw C fields.

import ctypes

import torch
import torch_npu

import triton
import triton.language as tl

DEVICE = "npu"


@tl.struct_type
class TensorMeta(ctypes.Structure):
    """A minimal device-side description of an input tensor."""

    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),  # pointer to the elements
        ("size", ctypes.c_int32),  # number of elements
        # [elements processed, number of launches] -- updated by the kernel.
        ("stats", ctypes.c_int32 * 2),
    ]

    def __init__(self, tensor=None, size=None):
        """Describe ``tensor``: anything with ``data_ptr()`` and ``numel()``.

        Called with no arguments the fields are left zeroed for manual filling;
        ``size`` overrides the element count taken from ``tensor``.
        """
        super().__init__()
        if tensor is not None:
            self.data = ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))
            self.size = tensor.numel() if size is None else size
        elif size is not None:
            self.size = size
        self.stats[0] = 0
        self.stats[1] = 0


# %%
# The kernel
# ----------
#
# Fields are read like attributes (``meta.size``) and array fields are indexed
# like arrays (``meta.stats[0]``).  ``meta.data`` is a pointer field: adding the
# offsets to it and loading yields the input block.
#
# Note the parameter is deliberately left unannotated: ``meta: TensorMeta`` does
# not work.  Triton evaluates an annotation as an expression, and a bare class
# name is a global variable, which kernels are not allowed to read.  Quoting it
# (``meta: "TensorMeta"``) is accepted but inert -- the parameter type comes
# from the value passed in (the ``StructView``), never from the annotation, so
# it would be documentation only.


@triton.jit
def scale_kernel(meta, out_ptr, alpha, BLOCK: tl.constexpr):
    n = meta.size  # scalar field read
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    values = tl.load(meta.data + offsets, mask=mask, other=0.0)  # deref a pointer field
    tl.store(out_ptr + offsets, values * alpha, mask=mask)
    meta.stats[0] = n  # array element write
    meta.stats[1] = meta.stats[1] + 1  # array element read + write


# %%
# Running it
# ----------
#
# ``n`` is chosen to match ``BLOCK`` so the example stays minimal; a real kernel
# would launch a grid and derive ``offsets`` from ``tl.program_id`` as usual.
# Note that the struct lives in device memory, so the fields the kernel writes
# are visible to the host once it copies the struct back.


def main():
    n = 8
    alpha = 2.0

    input_tensor = torch.arange(1, n + 1, dtype=torch.float32, device=DEVICE)
    output_tensor = torch.empty_like(input_tensor)

    # 1. Describe the input tensor (the constructor fills in the fields).
    meta = TensorMeta(input_tensor)

    # 2. Upload the struct and pass it to the kernel by reference.
    meta_dev = meta.to_device(DEVICE)
    scale_kernel[(1, )](meta_dev, output_tensor, alpha, BLOCK=n)
    torch.npu.synchronize()

    # 3. Copy the struct back and decode it with the same ctypes type.
    meta_back = TensorMeta.from_buffer(bytearray(meta_dev.buffer.cpu().numpy().tobytes()))

    expected = input_tensor * alpha
    torch.testing.assert_close(output_tensor, expected)
    assert meta_back.size == n
    assert meta_back.stats[0] == n, "kernel recorded the elements it processed"
    assert meta_back.stats[1] == 1, "kernel bumped the launch counter"

    print(f"input           : {input_tensor.cpu().tolist()}")
    print(f"output (x{alpha})   : {output_tensor.cpu().tolist()}")
    print(f"meta.size       : {meta_back.size}")
    print(f"meta.stats      : {[meta_back.stats[0], meta_back.stats[1]]}")
    print("OK")


if __name__ == "__main__":
    main()
