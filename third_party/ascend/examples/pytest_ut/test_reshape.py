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

@triton.jit
def fn_npu_(output_ptr, x_ptr,XB : tl.constexpr,YB : tl.constexpr,ZB : tl.constexpr):
    xidx=tl.arange(0,XB)
    yidx=tl.arange(0,YB)
    zidx=tl.arange(0,ZB)

    idx=xidx[:,None,None]*YB*ZB+yidx[None,:,None]*ZB+zidx[None,None,:]

    X = tl.load(x_ptr+idx)

    ret = tl.reshape(X,(ZB,XB*YB))

    oidx=tl.arange(0,ZB)[:,None]*XB*YB+tl.arange(0,XB*YB)[None,:]

    tl.store(output_ptr+oidx,ret)

testlist = [
    ('float32',torch.float32,2,256,16),
    ('float32',torch.float32,8,8,4),

    ('float16',torch.float16,2,256,16),
    ('float16',torch.float16,8,8,4),

    ('int8',torch.int8,2,256,16),
    ('int8',torch.int8,8,8,4),
]

@pytest.mark.parametrize('sigtype, dtype, XB, YB, ZB',testlist)
def test_ravel(sigtype, dtype, XB, YB, ZB):

    x = torch.randint(low=-128,high=128,size=(XB,YB,ZB),dtype=dtype).npu()
    ans = torch.reshape(x,(ZB,XB*YB))

    print(ans[0,0:16])

    output = torch.randint(1, (ZB,XB*YB), dtype=dtype).npu()

    fn_npu_[1,1,1](output, x, XB, YB, ZB)
    print(output[0,0:16])

    test_common.validate_cmp(sigtype,output,ans)