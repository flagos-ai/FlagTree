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


def torch_func(x0, x1):
    res = x0 // x1
    return res


@triton.jit
def triton_kernel(out_ptr0, in_ptr0, in_ptr1, N: tl.constexpr):
    idx = tl.arange(0, N)
    x = tl.load(in_ptr0 + idx)
    y = tl.load(in_ptr1 + idx)
    ret = x // y
    tl.store(out_ptr0 + idx, ret)


def triton_func(x0, x1, N):
    out = torch.empty_like(x0)
    triton_kernel[1, 1, 1](out, x0, x1, N)
    return out


types = [
    "int32",
    "bool",
]

shapes = [
    3,
    32,
    37,
    256,
    781,
]

@pytest.mark.parametrize("sigtype", types)
@pytest.mark.parametrize("N", shapes)
def test_floordiv(sigtype, N):
    x0 = test_common.generate_tensor(shape=(N,), dtype=sigtype).npu()
    x1 = test_common.generate_tensor(shape=(N,), dtype=sigtype).npu()
    x1 = x1.masked_fill(x1 == 0, 1)

    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1, N)
    test_common.validate_cmp(sigtype, triton_cal, torch_ref)

