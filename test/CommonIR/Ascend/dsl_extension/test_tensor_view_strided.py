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
def strided_view_load_store_kernel(src, dst, n_elements, BLOCK_SIZE: tl.constexpr, TRAVERSAL_STRIDE: tl.constexpr):
    block = tl.program_id(0)
    src_tiles = tle.make_strided_view(
        src,
        shape=[n_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
        traversal_strides=[TRAVERSAL_STRIDE],
    )
    dst_tiles = tle.make_strided_view(
        dst,
        shape=[n_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
        traversal_strides=[TRAVERSAL_STRIDE],
    )

    value = tle.tile_load(src_tiles, index=(block, ))
    tle.tile_store(dst_tiles, value=value, index=(block, ))


@triton.jit
def strided_view_padding_kernel(src, dst, src_elements, dst_elements, BLOCK_SIZE: tl.constexpr,
                                TRAVERSAL_STRIDE: tl.constexpr):
    block = tl.program_id(0)
    src_tiles = tle.make_strided_view(
        src,
        shape=[src_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
        traversal_strides=[TRAVERSAL_STRIDE],
        padding_value="inf",
    )
    dst_tiles = tle.make_strided_view(
        dst,
        shape=[dst_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
        traversal_strides=[TRAVERSAL_STRIDE],
    )

    value = tle.tile_load(src_tiles, index=(block, ))
    tle.tile_store(dst_tiles, value=value, index=(block, ))


def test_tensor_view_strided_load_store():
    grid_size = 4
    block_size = 1024
    traversal_stride = 1536
    n_elements = (grid_size - 1) * traversal_stride + block_size
    src = torch.rand(n_elements, device="npu")
    actual = torch.zeros_like(src)
    expected = torch.zeros_like(src)
    for block in range(grid_size):
        start = block * traversal_stride
        expected[start:start + block_size] = src[start:start + block_size]

    strided_view_load_store_kernel[(grid_size, )](src, actual, n_elements, BLOCK_SIZE=block_size,
                                                  TRAVERSAL_STRIDE=traversal_stride)

    torch.testing.assert_close(actual, expected)


def test_tensor_view_strided_padding():
    block_size = 4
    traversal_stride = 5
    src_elements = 13
    dst_elements = 14
    src = torch.arange(src_elements, dtype=torch.float32, device="npu")
    actual = torch.zeros(dst_elements, dtype=torch.float32, device="npu")

    strided_view_padding_kernel[(3, )](src, actual, src_elements, dst_elements, BLOCK_SIZE=block_size,
                                       TRAVERSAL_STRIDE=traversal_stride)

    expected = torch.zeros_like(actual)
    expected[0:4] = src[0:4]
    expected[5:9] = src[5:9]
    expected[10:13] = src[10:13]
    expected[13] = torch.inf
    torch.testing.assert_close(actual, expected)
