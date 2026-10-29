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

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit
def partition_view_load_store_kernel(src, dst, rows, cols, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    src_tiles = tle.make_partition_view(
        src,
        shape=[rows, cols],
        strides=[cols, 1],
        tile=[BLOCK_M, BLOCK_N],
    )
    dst_tiles = tle.make_partition_view(
        dst,
        shape=[rows, cols],
        strides=[cols, 1],
        tile=[BLOCK_M, BLOCK_N],
    )

    value = tle.tile_load(src_tiles, index=(row_block, col_block))
    tle.tile_store(dst_tiles, value=value, index=(row_block, col_block))


@triton.jit
def partition_view_padding_kernel(src, dst, src_elements, dst_elements, BLOCK_SIZE: tl.constexpr):
    block = tl.program_id(0)
    src_tiles = tle.make_partition_view(
        src,
        shape=[src_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
        padding_value="-inf",
    )
    dst_tiles = tle.make_partition_view(
        dst,
        shape=[dst_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
    )

    value = tle.tile_load(src_tiles, index=(block, ))
    tle.tile_store(dst_tiles, value=value, index=(block, ))


@triton.jit
def partition_view_scaled_store_kernel(src, scale, dst, rows, cols, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    row_block = tl.program_id(0)
    src_tiles = tle.make_partition_view(
        src,
        shape=[rows, cols],
        strides=[cols, 1],
        tile=[BLOCK_M, BLOCK_N],
    )
    dst_tiles = tle.make_partition_view(
        dst,
        shape=[rows, cols],
        strides=[cols, 1],
        tile=[BLOCK_M, BLOCK_N],
    )

    value = tle.tile_load(src_tiles, index=(row_block, 0))
    factor = tl.load(scale + tl.arange(0, BLOCK_N))
    tle.tile_store(dst_tiles, value=value * factor[None, :], index=(row_block, 0))


def test_tensor_view_partition_load_store():
    rows = 128
    cols = 128
    block_m = 64
    block_n = 64
    src = torch.rand((rows, cols), device="npu")
    actual = torch.zeros_like(src)

    grid = (triton.cdiv(rows, block_m), triton.cdiv(cols, block_n))
    partition_view_load_store_kernel[grid](src, actual, rows, cols, BLOCK_M=block_m, BLOCK_N=block_n)

    torch.testing.assert_close(actual, src)


def test_tensor_view_partition_padding():
    src_elements = 10
    block_size = 8
    dst_elements = 16
    src = torch.arange(src_elements, dtype=torch.float32, device="npu")
    actual = torch.empty(dst_elements, dtype=torch.float32, device="npu")

    partition_view_padding_kernel[(2, )](src, actual, src_elements, dst_elements, BLOCK_SIZE=block_size)

    expected = torch.full_like(actual, -torch.inf)
    expected[:src_elements] = src
    torch.testing.assert_close(actual, expected)


def test_tensor_view_partition_store_computed_value():
    rows = 61
    cols = 128
    block_m = 16
    padded_rows = triton.cdiv(rows, block_m) * block_m
    src = torch.rand((rows, cols), device="npu")
    scale = torch.rand((cols, ), device="npu")
    sentinel = -1.0
    actual = torch.full((padded_rows, cols), sentinel, device="npu")

    partition_view_scaled_store_kernel[(triton.cdiv(rows, block_m), )](src, scale, actual, rows, cols, BLOCK_M=block_m,
                                                                       BLOCK_N=cols)

    torch.testing.assert_close(actual[:rows], src * scale[None, :])
    assert torch.all(actual[rows:] == sentinel)
