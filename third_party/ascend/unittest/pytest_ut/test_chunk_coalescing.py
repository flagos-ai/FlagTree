# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

import pytest
import torch
import triton
import triton.language as tl

import test_common


@triton.jit
def kernel_chunk_coalescing_axis1(src, dst, N: tl.constexpr, BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    tile = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    base = batch * N
    values = tl.load(src + base + offsets, mask=mask, other=0.0)
    tl.store(dst + base + offsets, values, mask=mask)


@triton.jit
def kernel_chunk_coalescing_axis0(src, dst, N: tl.constexpr, BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    values = tl.load(src + offsets, mask=mask, other=0.0)
    tl.store(dst + offsets, values, mask=mask)


@triton.jit
def kernel_chunk_coalescing_axis2(src, dst, N: tl.constexpr, BLOCK: tl.constexpr):
    tile = tl.program_id(2)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    values = tl.load(src + offsets, mask=mask, other=0.0)
    tl.store(dst + offsets, values, mask=mask)


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_chunk_coalescing_grid_axis1_e2e(dtype):
    batch = 2
    block = 16
    num_tiles = 32
    n = block * num_tiles
    src = test_common.generate_tensor((batch, n), dtype).npu()
    dst = torch.empty_like(src)

    kernel_chunk_coalescing_axis1[(batch, num_tiles)](
        src,
        dst,
        n,
        BLOCK=block,
    )

    assert torch.equal(dst.cpu(), src.cpu())


def test_chunk_coalescing_tail_e2e():
    """A non-divisible tail remains a masked, conservative no-op candidate."""
    batch = 2
    block = 16
    num_tiles = 32
    n = block * num_tiles - 3
    src = test_common.generate_tensor((batch, n), "float32").npu()
    dst = torch.empty_like(src)

    kernel_chunk_coalescing_axis1[(batch, num_tiles)](
        src,
        dst,
        n,
        BLOCK=block,
    )

    assert torch.equal(dst.cpu(), src.cpu())


def test_chunk_coalescing_grid_axis0_e2e():
    block = 16
    num_tiles = 32
    n = block * num_tiles
    src = test_common.generate_tensor((n, ), "float32").npu()
    dst = torch.empty_like(src)

    kernel_chunk_coalescing_axis0[(num_tiles, )](
        src,
        dst,
        n,
        BLOCK=block,
    )

    assert torch.equal(dst.cpu(), src.cpu())


def test_chunk_coalescing_grid_axis2_e2e():
    block = 16
    num_tiles = 32
    n = block * num_tiles
    src = test_common.generate_tensor((n, ), "float32").npu()
    dst = torch.empty_like(src)

    kernel_chunk_coalescing_axis2[(1, 1, num_tiles)](
        src,
        dst,
        n,
        BLOCK=block,
    )

    assert torch.equal(dst.cpu(), src.cpu())
