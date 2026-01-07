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
import numpy as np

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(parent_dir)
import test_common

# PR: Pointiwise-Reduction pattern, reduction in last axis
def numpy_sum_pr(x0, x1):
    res = np.sum(x0, axis=-1) + x1
    return res

@triton.jit
def triton_sum_pr(out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, XBLOCK_SUB : tl.constexpr, RBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)
        xmask = xindex[:,None] < xnumel
        x0 = xindex
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (RBLOCK*x0[:, None])), xmask & rmask)
        tmp4 = tl.load(in_ptr1 + (x0), xindex < xnumel)
        tmp1 = tl.reshape(tmp0, [XBLOCK_SUB, RBLOCK])
        tmp3 = tl.sum(tmp1, 1)
        tmp5 = tmp3 + tmp4
        tl.store(out_ptr0 + (xindex), tmp5, None)


# fp16 use numpy
@pytest.mark.parametrize('param_list',
                         [
                            # ['float32', (8, 8, 4), 8, 2],
                            ['float32', (8, 8, 64), 8, 2],
                            ['float32', (8, 8, 512), 8, 2],
                            ['float16', (8, 8, 4), 8, 2],
                            ['float16', (8, 8, 64), 8, 2],
                            ['float16', (8, 8, 512), 8, 2],
                            ['int8', (8, 8, 4), 8, 2],
                            ['int8', (8, 8, 64), 8, 2],
                            ['int8', (8, 8, 512), 8, 2],
                         ]
                         )

def test_sum_pr(param_list):
    dtype, shape, ncore, xblock_sub = param_list
    import math
    numel = math.prod(shape)
    xblock = numel // shape[-1] // ncore
    rblock = shape[-1]
    assert(ncore * xblock * shape[-1] == numel)
    xn1 = np.random.randn(shape[0], shape[1], shape[2]).astype(eval('np.' + dtype))
    xn2 = np.random.randn(shape[0], shape[1]).astype(eval('np.' + dtype))
    x0 = torch.tensor(xn1).npu()
    x1 = torch.tensor(xn2).npu()
    if dtype == 'int8':
        y_ref = numpy_sum_pr(xn1, xn2).astype(np.int8)
    else:
        y_ref = numpy_sum_pr(xn1, xn2)
    y_cal = test_common.generate_tensor(shape[:-1], dtype).npu()
    triton_sum_pr[ncore,1,1](y_cal, x0, x1, x1.numel(), rblock, xblock, xblock_sub, rblock)
    test_common.validate_cmp(dtype, y_cal, torch.tensor(y_ref).npu())