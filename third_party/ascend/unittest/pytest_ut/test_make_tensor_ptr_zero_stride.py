# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

import torch
import torch_npu

import triton
import triton.language as tl
import pytest


@triton.jit
def broadcast_load_ref_kernel(in_ptr, out_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Reference: load a (BLOCK_M, BLOCK_N, BLOCK_K) block where every element
    is ``in_ptr[0]`` (broadcast of a single scalar to the block shape)."""
    off_m = tl.arange(0, BLOCK_M)[:, None, None]
    off_n = tl.arange(0, BLOCK_N)[None, :, None]
    off_k = tl.arange(0, BLOCK_K)[None, None, :]
    val = tl.load(in_ptr + off_m * 0 + off_n * 0 + off_k * 0)
    tl.store(out_ptr + off_m * (BLOCK_N * BLOCK_K) + off_n * BLOCK_K + off_k, val)


@triton.jit
def broadcast_store_ref_kernel(in_ptr, out_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Reference: store ``in_ptr[0] + 1`` into every position of a
    (BLOCK_M, BLOCK_N, BLOCK_K) block. Under broadcast-zero-stride semantics
    every store site receives the same scalar; the last writer wins on the
    single shared address."""
    off_m = tl.arange(0, BLOCK_M)[:, None, None]
    off_n = tl.arange(0, BLOCK_N)[None, :, None]
    off_k = tl.arange(0, BLOCK_K)[None, None, :]
    val = tl.load(in_ptr) + 1.0
    tl.store(out_ptr + off_m * (BLOCK_N * BLOCK_K) + off_n * BLOCK_K + off_k, val)


@triton.jit
def broadcast_advance_ref_kernel(in_ptr, out_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Reference: advance with strides (0, 0, 0) keeps the address at
    ``base``, so the loaded block is still ``in_ptr[0]`` broadcast."""
    off_m = tl.arange(0, BLOCK_M)[:, None, None]
    off_n = tl.arange(0, BLOCK_N)[None, :, None]
    off_k = tl.arange(0, BLOCK_K)[None, None, :]
    val = tl.load(in_ptr)
    tl.store(out_ptr + off_m * (BLOCK_N * BLOCK_K) + off_n * BLOCK_K + off_k, val)


@triton.jit
def broadcast_load_kernel(in_ptr, out_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Use ``make_block_ptr`` with all-zero strides for the load."""
    bptr = tl.make_block_ptr(
        base=in_ptr,
        shape=(BLOCK_M, BLOCK_N, BLOCK_K),
        strides=(0, 0, 0),
        offsets=(0, 0, 0),
        block_shape=(BLOCK_M, BLOCK_N, BLOCK_K),
        order=(0, 1, 2),
    )
    val = tl.load(bptr, boundary_check=(0, 1, 2))

    out = tl.make_block_ptr(
        base=out_ptr,
        shape=(BLOCK_M, BLOCK_N, BLOCK_K),
        strides=(BLOCK_N * BLOCK_K, BLOCK_K, 1),
        offsets=(0, 0, 0),
        block_shape=(BLOCK_M, BLOCK_N, BLOCK_K),
        order=(0, 1, 2),
    )
    tl.store(out, val, boundary_check=(0, 1, 2))


@triton.jit
def broadcast_store_kernel(in_ptr, out_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Use ``make_block_ptr`` with all-zero strides for the store."""
    scalar = tl.load(in_ptr)

    bptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(BLOCK_M, BLOCK_N, BLOCK_K),
        strides=(0, 0, 0),
        offsets=(0, 0, 0),
        block_shape=(BLOCK_M, BLOCK_N, BLOCK_K),
        order=(0, 1, 2),
    )
    tl.store(bptr, scalar + 1.0, boundary_check=(0, 1, 2))


@triton.jit
def broadcast_advance_kernel(in_ptr, out_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Use ``make_block_ptr`` with all-zero strides followed by ``tl.advance``
    with non-zero offsets — the advanced pointer must still alias base."""
    bptr = tl.make_block_ptr(
        base=in_ptr,
        shape=(BLOCK_M, BLOCK_N, BLOCK_K),
        strides=(0, 0, 0),
        offsets=(0, 0, 0),
        block_shape=(BLOCK_M, BLOCK_N, BLOCK_K),
        order=(0, 1, 2),
    )
    advanced = tl.advance(bptr, [2, 3, 4])
    val = tl.load(advanced, boundary_check=(0, 1, 2))

    out = tl.make_block_ptr(
        base=out_ptr,
        shape=(BLOCK_M, BLOCK_N, BLOCK_K),
        strides=(BLOCK_N * BLOCK_K, BLOCK_K, 1),
        offsets=(0, 0, 0),
        block_shape=(BLOCK_M, BLOCK_N, BLOCK_K),
        order=(0, 1, 2),
    )
    tl.store(out, val, boundary_check=(0, 1, 2))


def _check_close(actual: torch.Tensor, expected: torch.Tensor, tag: str):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=0,
        atol=0,
        msg=(f"{tag}: broadcast-zero-stride kernel diverged from the "
             "reference implementation"),
    )


def test_zero_stride_make_block_ptr_load():
    BLOCK_M, BLOCK_N, BLOCK_K = 4, 16, 16

    scalar = torch.tensor([3.14], dtype=torch.float32, device="npu")

    out_kernel = torch.zeros((BLOCK_M, BLOCK_N, BLOCK_K), dtype=torch.float32, device="npu")
    out_ref = torch.zeros_like(out_kernel)

    broadcast_load_kernel[(1, )](scalar, out_kernel, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    broadcast_load_ref_kernel[(1, )](scalar, out_ref, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    torch.npu.synchronize()

    _check_close(out_kernel, out_ref, "load")


@pytest.mark.skip(reason="waiting for fix")
def test_zero_stride_make_block_ptr_store():

    BLOCK_M, BLOCK_N, BLOCK_K = 4, 16, 16

    scalar = torch.tensor([2.0], dtype=torch.float32, device="npu")
    sink = torch.tensor([0.0], dtype=torch.float32, device="npu")
    sink_ref = torch.tensor([0.0], dtype=torch.float32, device="npu")

    broadcast_store_kernel[(1, )](scalar, sink, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    broadcast_store_ref_kernel[(1, )](scalar, sink_ref, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    torch.npu.synchronize()

    _check_close(sink, sink_ref, "store")


@pytest.mark.skip(reason="waiting for fix")
def test_zero_stride_make_block_ptr_advance():

    BLOCK_M, BLOCK_N, BLOCK_K = 4, 16, 16

    scalar = torch.tensor([1.5], dtype=torch.float32, device="npu")

    out_kernel = torch.zeros((BLOCK_M, BLOCK_N, BLOCK_K), dtype=torch.float32, device="npu")
    out_ref = torch.zeros_like(out_kernel)

    broadcast_advance_kernel[(1, )](scalar, out_kernel, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    broadcast_advance_ref_kernel[(1, )](scalar, out_ref, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    torch.npu.synchronize()

    _check_close(out_kernel, out_ref, "advance")
