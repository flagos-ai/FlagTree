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
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit
def gather_view_load_kernel(src, indices, dst, n_elements, BLOCK_SIZE: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    data_indices = tl.load(indices + offsets)
    src_gather = tle.make_gather_scatter_view(
        src,
        shape=[n_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
        sparse_dim=[0],
    )
    dst_tiles = tle.make_partition_view(
        dst,
        shape=[n_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
    )

    value = tle.tile_load(src_gather, index=(data_indices, ))
    tle.tile_store(dst_tiles, value=value, index=(block, ))


@triton.jit
def scatter_view_store_kernel(src, indices, dst, n_elements, BLOCK_SIZE: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    data_indices = tl.load(indices + offsets)
    src_tiles = tle.make_partition_view(
        src,
        shape=[n_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
    )
    dst_scatter = tle.make_gather_scatter_view(
        dst,
        shape=[n_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
        sparse_dim=[0],
    )

    value = tle.tile_load(src_tiles, index=(block, ))
    tle.tile_store(dst_scatter, value=value, index=(data_indices, ))


@triton.jit
def gather_view_padding_kernel(src, indices, dst, src_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    data_indices = tl.load(indices + offsets)
    src_gather = tle.make_gather_scatter_view(
        src,
        shape=[src_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
        sparse_dim=[0],
        padding_value="nan",
        bounds_check=True,
    )
    dst_tiles = tle.make_partition_view(
        dst,
        shape=[BLOCK_SIZE],
        strides=[1],
        tile=[BLOCK_SIZE],
    )

    value = tle.tile_load(src_gather, index=(data_indices, ))
    tle.tile_store(dst_tiles, value=value, index=(0, ))


@triton.jit
def scatter_view_bounds_kernel(src, indices, dst, dst_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    data_indices = tl.load(indices + offsets)
    src_tiles = tle.make_partition_view(
        src,
        shape=[BLOCK_SIZE],
        strides=[1],
        tile=[BLOCK_SIZE],
    )
    dst_scatter = tle.make_gather_scatter_view(
        dst,
        shape=[dst_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
        sparse_dim=[0],
        bounds_check=True,
    )

    value = tle.tile_load(src_tiles, index=(0, ))
    tle.tile_store(dst_scatter, value=value, index=(data_indices, ))


@triton.jit
def gather_unbounded_kernel(src, indices, dst, num_indices, WIDTH: tl.constexpr, BLOCK_T: tl.constexpr):
    block = tl.program_id(0)
    rows = tl.load(indices + block * BLOCK_T + tl.arange(0, BLOCK_T))
    src_gather = tle.make_gather_scatter_view(
        src,
        shape=[None, WIDTH],
        strides=[WIDTH, 1],
        tile=[BLOCK_T, WIDTH],
        sparse_dim=[0],
    )
    dst_tiles = tle.make_partition_view(
        dst,
        shape=[num_indices, WIDTH],
        strides=[WIDTH, 1],
        tile=[BLOCK_T, WIDTH],
    )

    value = tle.tile_load(src_gather, index=(rows, 0))
    tle.tile_store(dst_tiles, value=value, index=(block, 0))


@triton.jit
def scatter_unbounded_kernel(src, indices, dst, num_indices, WIDTH: tl.constexpr, BLOCK_T: tl.constexpr):
    block = tl.program_id(0)
    rows = tl.load(indices + block * BLOCK_T + tl.arange(0, BLOCK_T))
    src_tiles = tle.make_partition_view(
        src,
        shape=[num_indices, WIDTH],
        strides=[WIDTH, 1],
        tile=[BLOCK_T, WIDTH],
    )
    dst_scatter = tle.make_gather_scatter_view(
        dst,
        shape=[None, WIDTH],
        strides=[WIDTH, 1],
        tile=[BLOCK_T, WIDTH],
        sparse_dim=[0],
    )

    value = tle.tile_load(src_tiles, index=(block, 0))
    tle.tile_store(dst_scatter, value=value, index=(rows, 0))


@triton.jit
def sparse_extent_required_kernel(src, indices, dst, WIDTH: tl.constexpr, BLOCK_T: tl.constexpr):
    rows = tl.load(indices + tl.arange(0, BLOCK_T))
    src_gather = tle.make_gather_scatter_view(
        src,
        shape=[None, WIDTH],
        strides=[WIDTH, 1],
        tile=[BLOCK_T, WIDTH],
        sparse_dim=[0],
        bounds_check=True,
    )
    value = tle.tile_load(src_gather, index=(rows, 0))
    tl.store(dst + tl.arange(0, BLOCK_T)[:, None] * WIDTH + tl.arange(0, WIDTH)[None, :], value)


@triton.jit
def dense_extent_required_kernel(src, indices, dst, WIDTH: tl.constexpr, BLOCK_T: tl.constexpr):
    rows = tl.load(indices + tl.arange(0, BLOCK_T))
    src_gather = tle.make_gather_scatter_view(
        src,
        shape=[BLOCK_T, None],
        strides=[WIDTH, 1],
        tile=[BLOCK_T, WIDTH],
        sparse_dim=[0],
    )
    value = tle.tile_load(src_gather, index=(rows, 0))
    tl.store(dst + tl.arange(0, BLOCK_T)[:, None] * WIDTH + tl.arange(0, WIDTH)[None, :], value)


@triton.jit
def scatter_masked_rows_kernel(src, dst, row_sum, num_rows, WIDTH: tl.constexpr, BLOCK_T: tl.constexpr,
                               BOUNDS_CHECK: tl.constexpr):
    block = tl.program_id(0)
    src_tiles = tle.make_partition_view(
        src,
        shape=[num_rows, WIDTH],
        strides=[WIDTH, 1],
        tile=[BLOCK_T, WIDTH],
    )
    value = tle.tile_load(src_tiles, index=(block, 0))
    rows = block * BLOCK_T + tl.arange(0, BLOCK_T)
    masked_rows = tl.where(rows < num_rows, rows, -1)
    dst_scatter = tle.make_gather_scatter_view(
        dst,
        shape=[num_rows, WIDTH],
        strides=[WIDTH, 1],
        tile=[BLOCK_T, WIDTH],
        sparse_dim=[0],
        bounds_check=BOUNDS_CHECK,
    )
    tle.tile_store(dst_scatter, value=value, index=(masked_rows, 0))
    tl.store(row_sum + rows, tl.sum(value.to(tl.float32), axis=1))


def test_tensor_view_gather_load():
    n_elements = 1024
    block_size = 256
    src = torch.rand(n_elements, device="npu")
    indices = torch.randint(0, n_elements, (n_elements, ), dtype=torch.int32, device="npu")
    actual = torch.empty_like(src)

    gather_view_load_kernel[(triton.cdiv(n_elements, block_size), )](src, indices, actual, n_elements,
                                                                     BLOCK_SIZE=block_size)

    torch.testing.assert_close(actual, src[indices.long()])


def test_tensor_view_scatter_store():
    n_elements = 1024
    block_size = 256
    src = torch.rand(n_elements, device="npu")
    indices = torch.randperm(n_elements, device="npu").to(torch.int32)
    actual = torch.zeros_like(src)
    expected = torch.zeros_like(src)
    expected[indices.long()] = src

    scatter_view_store_kernel[(triton.cdiv(n_elements, block_size), )](src, indices, actual, n_elements,
                                                                       BLOCK_SIZE=block_size)

    torch.testing.assert_close(actual, expected)


def test_tensor_view_gather_padding():
    src = torch.arange(8, dtype=torch.float32, device="npu")
    indices = torch.tensor([0, 7, -1, 8], dtype=torch.int32, device="npu")
    actual = torch.empty(4, dtype=torch.float32, device="npu")

    gather_view_padding_kernel[(1, )](src, indices, actual, src.numel(), BLOCK_SIZE=4)

    expected = torch.tensor([0, 7, torch.nan, torch.nan], dtype=torch.float32, device="npu")
    torch.testing.assert_close(actual, expected, equal_nan=True)


def test_tensor_view_scatter_bounds():
    src = torch.tensor([11, 23, 45, 67], dtype=torch.float32, device="npu")
    indices = torch.tensor([-1, 0, 7, 8], dtype=torch.int32, device="npu")
    actual = torch.zeros(8, dtype=torch.float32, device="npu")

    scatter_view_bounds_kernel[(1, )](src, indices, actual, actual.numel(), BLOCK_SIZE=4)

    expected = torch.zeros_like(actual)
    expected[0] = src[1]
    expected[7] = src[2]
    torch.testing.assert_close(actual, expected)


def test_tensor_view_gather_unbounded_extent():
    num_rows, width, block_t = 64, 32, 16
    src = torch.rand((num_rows, width), device="npu")
    indices = torch.randint(0, num_rows, (num_rows, ), dtype=torch.int32, device="npu")
    actual = torch.empty_like(src)

    gather_unbounded_kernel[(num_rows // block_t, )](src, indices, actual, num_rows, WIDTH=width, BLOCK_T=block_t)

    torch.testing.assert_close(actual, src[indices.long()])


def test_tensor_view_scatter_unbounded_extent():
    num_rows, width, block_t = 64, 32, 16
    src = torch.rand((num_rows, width), device="npu")
    indices = torch.randperm(num_rows, device="npu").to(torch.int32)
    actual = torch.zeros_like(src)
    expected = torch.zeros_like(src)
    expected[indices.long()] = src

    scatter_unbounded_kernel[(num_rows // block_t, )](src, indices, actual, num_rows, WIDTH=width, BLOCK_T=block_t)

    torch.testing.assert_close(actual, expected)


def test_tensor_view_bounds_check_requires_sparse_extent():
    width, block_t = 32, 16
    src = torch.rand((block_t, width), device="npu")
    indices = torch.arange(block_t, dtype=torch.int32, device="npu")
    actual = torch.empty_like(src)

    with pytest.raises(triton.compiler.errors.CompilationError,
                       match="Expected `shape\\[0\\]` to be given when `bounds_check` is enabled"):
        sparse_extent_required_kernel[(1, )](src, indices, actual, WIDTH=width, BLOCK_T=block_t)


def test_tensor_view_dense_extent_required():
    width, block_t = 32, 16
    src = torch.rand((block_t, width), device="npu")
    indices = torch.arange(block_t, dtype=torch.int32, device="npu")
    actual = torch.empty_like(src)

    with pytest.raises(triton.compiler.errors.CompilationError,
                       match="Expected `shape\\[1\\]` to be given: only sparse dimensions may omit it"):
        dense_extent_required_kernel[(1, )](src, indices, actual, WIDTH=width, BLOCK_T=block_t)


@pytest.mark.parametrize("bounds_check", [False, True])
def test_tensor_view_scatter_masked_rows_shared_index(bounds_check):
    num_rows, width, block_t = 64, 128, 8
    src = torch.rand((num_rows, width), dtype=torch.float16, device="npu")
    dst = torch.zeros_like(src)
    row_sum = torch.zeros(num_rows, dtype=torch.float32, device="npu")

    scatter_masked_rows_kernel[(num_rows // block_t, )](src, dst, row_sum, num_rows, WIDTH=width, BLOCK_T=block_t,
                                                        BOUNDS_CHECK=bounds_check)

    torch.testing.assert_close(dst, src)
    torch.testing.assert_close(row_sum, src.to(torch.float32).sum(dim=1))
