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

import triton
import triton.language as tl

import torch
import torch_npu
import pytest
import test_common
import math


@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr, XB: tl.constexpr):

    idx = tl.arange(0, XB)
    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)

    ret = tl.cat(X, Y, can_reorder=True)

    oidx = tl.arange(0, XB * 2)

    tl.store(output_ptr + oidx, ret)


# The CAT operator in the Triton community also does not support boolean types.
@pytest.mark.parametrize('shape', [(32, ), (741, )])  #triton only support 1D cat
@pytest.mark.parametrize('dtype', [
    'float32',
])
def test_cat(shape, dtype):
    m = shape[0]
    x = torch.full((m, ), 100, dtype=eval("torch." + dtype)).npu()
    y = torch.full((m, ), 30, dtype=eval("torch." + dtype)).npu()

    output = torch.randint(1, (m * 2, ), dtype=eval("torch." + dtype)).npu()

    ans = torch.cat((x, y), dim=0)

    fn_npu_[1, 1, 1](output, x, y, m)

    test_common.validate_cmp(dtype, ans, output)


@triton.jit
def cat_pointer_load_kernel(
    left_ptr,
    right_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    MASKED: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    pointers = tl.cat(left_ptr + offsets, right_ptr + offsets, can_reorder=True)
    output_offsets = tl.arange(0, BLOCK_SIZE * 2)

    if MASKED:
        mask = (output_offsets != 2) & (output_offsets != BLOCK_SIZE + 3)
        other = output_offsets.to(tl.float32) + 1000.0
        values = tl.load(pointers, mask=mask, other=other)
    else:
        values = tl.load(pointers)

    tl.store(output_ptr + output_offsets, values)


@triton.jit
def cat_pointer_store_kernel(
    left_ptr,
    right_ptr,
    BLOCK_SIZE: tl.constexpr,
    MASKED: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    pointers = tl.cat(left_ptr + offsets, right_ptr + offsets, can_reorder=True)
    input_offsets = tl.arange(0, BLOCK_SIZE * 2)
    values = input_offsets.to(tl.float32) + 200.0

    if MASKED:
        mask = (input_offsets != 1) & (input_offsets != BLOCK_SIZE + 5)
        tl.store(pointers, values, mask=mask)
    else:
        tl.store(pointers, values)


@pytest.mark.parametrize("masked", [False, True], ids=["unmasked", "masked_tensor_other"])
def test_cat_pointer_load(masked):
    block_size = 16
    total_size = block_size * 2
    left = torch.arange(block_size, dtype=torch.float32, device="npu") + 10.0
    right = torch.arange(block_size, dtype=torch.float32, device="npu") + 100.0
    output = torch.empty(total_size, dtype=torch.float32, device="npu")

    cat_pointer_load_kernel[(1, )](left, right, output, BLOCK_SIZE=block_size, MASKED=masked)

    expected = torch.cat((left, right))
    if masked:
        offsets = torch.arange(total_size, dtype=torch.int32, device="npu")
        mask = (offsets != 2) & (offsets != block_size + 3)
        other = offsets.to(torch.float32) + 1000.0
        expected = torch.where(mask, expected, other)

    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("masked", [False, True], ids=["unmasked", "masked"])
def test_cat_pointer_store(masked):
    block_size = 16
    total_size = block_size * 2
    sentinel = -1.0
    left = torch.full((block_size, ), sentinel, dtype=torch.float32, device="npu")
    right = torch.full((block_size, ), sentinel, dtype=torch.float32, device="npu")

    cat_pointer_store_kernel[(1, )](left, right, BLOCK_SIZE=block_size, MASKED=masked)

    offsets = torch.arange(total_size, dtype=torch.int32, device="npu")
    values = offsets.to(torch.float32) + 200.0
    if masked:
        mask = (offsets != 1) & (offsets != block_size + 5)
        values = torch.where(mask, values, torch.full_like(values, sentinel))

    torch.testing.assert_close(left, values[:block_size], rtol=0, atol=0)
    torch.testing.assert_close(right, values[block_size:], rtol=0, atol=0)
