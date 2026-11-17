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

import os

import pytest
import torch
import torch_npu
import triton
import triton.backends.ascend.runtime
import triton.language as tl

os.environ['TRITON_PRINT_AUTOTUNING'] = '0'


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'XBLOCK_SUB': 32}),
        triton.Config({'XBLOCK': 128, 'XBLOCK_SUB': 64}),
        triton.Config({'XBLOCK': 396, 'XBLOCK_SUB': 6}),
    ], key=["n_rows", "n_cols"], hints={
        "auto_gen_config": False,
    })
@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    XBLOCK: tl.constexpr,
    XBLOCK_SUB: tl.constexpr,
):
    row_start = tl.program_id(0) * XBLOCK
    for row_idx in tl.range(0, XBLOCK, XBLOCK_SUB):
        row_offsets = row_start + row_idx + tl.arange(0, XBLOCK_SUB)[:, None]
        col_offsets = tl.arange(0, BLOCK_SIZE)[None, :]
        xmask = row_offsets < n_rows
        ymask = col_offsets < n_cols
        mask = xmask & ymask
        input_ptrs = input_ptr + (row_offsets * input_row_stride + col_offsets)
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        row_minus_max = row - tl.max(row, axis=1).reshape(XBLOCK_SUB, 1).broadcast_to(XBLOCK_SUB, BLOCK_SIZE)
        numerator = tl.exp(row_minus_max)
        denominator = (tl.sum(numerator, axis=1).reshape(XBLOCK_SUB, 1).broadcast_to(XBLOCK_SUB, BLOCK_SIZE))
        softmax_output = numerator / denominator
        output_ptrs = output_ptr + (row_offsets * output_row_stride + col_offsets)
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax_torch(x):
    return torch.softmax(x, axis=-1)


def softmax_autotune(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = n_cols
    y = torch.empty_like(x)
    softmax_kernel[lambda meta: (triton.cdiv(n_rows, meta["XBLOCK"]), 1, 1)](y, x, x.stride(0), y.stride(0), n_rows,
                                                                             n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return y


@pytest.mark.autotune
@pytest.mark.parametrize('shape,dtype', [
    ((16896, 1024), torch.float32),
])
def test_softmax(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device="npu")
    y_torch = softmax_torch(x)
    y_triton = softmax_autotune(x)
    torch.testing.assert_close(y_torch, y_triton, rtol=1e-03, atol=1e-03, equal_nan=True)
