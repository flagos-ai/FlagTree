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
import triton
import triton.language as tl
import torch_npu
import pytest


@triton.jit
def swap_kernel(
    x_ptr,  # *Pointer* to first inout vector.
    y_ptr,  # *Pointer* to second inout vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    # NOTE: `constexpr` so it can be used as a shape value.
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    tl.store(x_ptr + offsets, y)
    tl.store(y_ptr + offsets, x)



def swap(x: torch.Tensor, y: torch.Tensor, size):
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    swap_kernel[grid](x, y, BLOCK_SIZE=size)


@pytest.mark.parametrize('shape', [(1,), (3,), (4,), (7,), (8,), (11,), (16,), (512,), (1024,)])
def test(shape):
    x = torch.rand(shape).npu()
    y = torch.rand(shape).npu()
    assert not torch.equal(x, y)
    x_ = x.clone()
    y_ = y.clone()
    swap(x, y, shape[0])
    torch.testing.assert_close(x, y_, rtol=1e-04, atol=1e-04, equal_nan=True)
    torch.testing.assert_close(y, x_, rtol=1e-04, atol=1e-04, equal_nan=True)