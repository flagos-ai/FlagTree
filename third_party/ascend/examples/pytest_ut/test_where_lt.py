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


def torch_where_lt_case1(x0, x1):
    res = torch.where(x0 < x1, x0, 1)
    return res

@triton.jit
def triton_where_lt_case1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tmp1 = tl.load(in_ptr1 + (x0), xmask)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.where(tmp2, tmp0, 1)
        tl.store(out_ptr0 + (xindex), tmp3, xmask)

@pytest.mark.parametrize('param_list',
                         [
                            ['float32', (2, 1024, 8), 2, 8192, 1024],
                            ['float16', (2, 1024, 8), 2, 8192, 1024],
                            ['int8', (2, 1024, 8), 2, 8192, 1024],
                         ]
                         )

def test_where_lt_case1(param_list):
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    x1 = test_common.generate_tensor(shape, dtype).npu()
    y_ref = torch_where_lt_case1(x0, x1)
    y_cal = test_common.generate_tensor(shape, dtype).npu()
    triton_where_lt_case1[ncore, 1, 1](x0, x1, y_cal, x0.numel(), xblock, xblock_sub)
    test_common.validate_cmp(dtype, y_cal, y_ref)