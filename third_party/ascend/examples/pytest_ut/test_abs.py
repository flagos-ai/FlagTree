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
import numpy as np
import torch
import pytest
import test_common

def torch_pointwise(x0):
    if x0.dtype in [torch.uint8, torch.uint16, torch.uint32, torch.uint64]:
        return x0
    res = torch.abs(x0)
    return res


@triton.jit
def triton_abs(in_ptr0, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop1 in range(loops1):
        x0_prime = offset + (loop1 * XBLOCK_SUB) + base1
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp2 = tl.abs(tmp0)
        tl.store(out_ptr0 + (x0), tmp2, None)


@pytest.mark.parametrize('param_list',
                         [
                             ['float16', (2, 4096, 8), 32, 2048, 64],
                            #  ['bfloat16', (2, 4096, 8), 32, 2048, 64],
                             ['float32', (2, 4096, 8), 32, 2048, 64],
                             ['int8', (2, 4096, 8), 32, 2048, 64],
                            #  ['int16', (2, 4096, 8), 32, 2048, 64],
                            #  ['int32', (2, 4096, 8), 32, 2048, 64],
                            #  ['int64', (2, 4096, 8), 32, 2048, 64],
                             ['uint8', (2, 4096, 8), 32, 2048, 64],
                            #  ['uint16', (2, 4096, 8), 32, 2048, 64],
                            #  ['uint32', (2, 4096, 8), 32, 2048, 64],
                            #  ['uint64', (2, 4096, 8), 32, 2048, 64],
                         ]
                         )

def test_case(param_list):
    dtype, shape, ncore, xblock, xblock_sub = param_list
    np_x0 = test_common.generate_numpy(shape, dtype)
    x0 = torch.from_numpy(np_x0).to(eval('torch.' + dtype)).npu()
    y_ref = torch_pointwise(x0)
    y_cal = torch.zeros(shape, dtype = eval('torch.' + dtype)).npu()
    triton_abs[ncore, 1, 1](x0, y_cal, xblock, xblock_sub)
    test_common.validate_cmp(dtype, y_cal, y_ref)
