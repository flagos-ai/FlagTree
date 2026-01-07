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
import pytest
import test_common
from test_common import TestUtils
import math
import triton.language.extra.ascend.libdevice as libdevice
def torch_pointwise(x0):
    res = torch.tan(x0)
    return res

@triton.jit
def triton_tan(in_ptr0, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop1 in range(loops1):
        x0_prime = offset + (loop1 * XBLOCK_SUB) + base1
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp2 = libdevice.tan(tmp0)
        tl.store(out_ptr0 + (x0), tmp2, None)

@pytest.mark.parametrize('shape', TestUtils.test_shape1d)
@pytest.mark.parametrize('dtype', ['float32', 'float16'])
def test_case(dtype, shape):
    x0 = test_common.generate_tensor(shape, dtype).npu()

    numel = x0.numel()
    ncore = 1 if numel <= 32 else 32
    xblock = math.ceil(numel / ncore)
    xblock_sub = numel if numel <= ncore else math.ceil(numel / ncore)

    y_ref = torch_pointwise(x0)
    y_cal = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_tan[ncore, 1, 1](x0, y_cal, xblock, xblock_sub)
    test_common.validate_cmp(dtype, y_cal, y_ref)
