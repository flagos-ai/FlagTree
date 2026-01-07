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
import torch_npu
import triton
import triton.language as tl
from triton.runtime.libentry import libentry
import pytest
from test_common import generate_tensor, validate_cmp, _32bit_dtypes, _16bit_dtypes


def torch_func(x0):
    return torch.sum(x0)


@pytest.mark.parametrize("dtype", _32bit_dtypes)
@pytest.mark.parametrize("shape", [(1,), (3,), (8,), (37,), (64,), (781,)])
def test_sum(dtype, shape):

    @libentry()
    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, XBLOCK: tl.constexpr):
        idx = tl.arange(0, XBLOCK)
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = tl.sum(tmp0)
        tl.store(out_ptr0 + idx, tmp1)

    def triton_func(x0):
        out = x0[0]
        triton_kernel[1, 1, 1](out, x0, x0.numel())
        return out

    x0 = generate_tensor(shape=shape, dtype=dtype).npu()
    torch_ref = torch_func(x0)
    triton_cal = triton_func(x0)
    validate_cmp(dtype, torch_ref, triton_cal)


@triton.jit
def _reduce_combine(a, b):
    return a + b


@pytest.mark.parametrize("dtype", _32bit_dtypes)
@pytest.mark.parametrize("shape", [(1,), (3,), (8,), (37,), (64,), (781,)])
def test_reduce_sum(dtype, shape):

    @libentry()
    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, XBLOCK: tl.constexpr):
        idx = tl.arange(0, XBLOCK)
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = tl.reduce(tmp0, 0, _reduce_combine)
        tl.store(out_ptr0 + idx, tmp1)

    def triton_func(x0):
        out = x0[0]
        triton_kernel[1, 1, 1](out, x0, x0.numel())
        return out

    x0 = generate_tensor(shape=shape, dtype=dtype).npu()
    torch_ref = torch_func(x0)
    triton_cal = triton_func(x0)
    validate_cmp(dtype, torch_ref, triton_cal)
