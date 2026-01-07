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

import triton
import triton.language as tl
import triton.language.extra.ascend.libdevice as libdevice
import test_common

import torch
import torch_npu

@triton.jit
def triton_sinh(in_ptr0, out_ptr0, XBLOCK : tl.constexpr, XBLOCK_SUB : tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop1 in range(loops1):
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = libdevice.sinh(tmp0)
        tl.store(out_ptr0 + (x0), tmp1, None)


@pytest.mark.parametrize('param_list',
                            [
                                'float32',
                                'float16',
                                'bfloat16'
                            ])
def test_sinh_special(param_list):
    dtype = param_list
    x_near_zero = torch.linspace(-1.0, 1.0, 256, dtype=eval("torch."+dtype)).npu()
    x_medium_neg = torch.linspace(-10.0, -1.0, 128, dtype=eval("torch."+dtype)).npu()
    x_medium_pos = torch.linspace(1.0, 10.0, 128, dtype=eval("torch."+dtype)).npu()

    x_large_neg = torch.linspace(-1e4, -10.0, 64, dtype=eval("torch."+dtype)).npu()
    x_large_pos = torch.linspace(10.0, 1e4, 64, dtype=eval("torch."+dtype)).npu()

    x0 = torch.cat([x_near_zero, x_medium_neg, x_medium_pos, x_large_neg, x_large_pos], dim=0)
    
    y_ref = torch.sinh(x0)
    y_cal = torch.zeros_like(y_ref)
    triton_sinh[1, 1, 1](x0, y_cal, x0.shape[0], x0.shape[0])
    test_common.validate_cmp(dtype, y_cal, y_ref)