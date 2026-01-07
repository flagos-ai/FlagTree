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
def fn_npu_(output_ptr, x_ptr,output_ptr1,XB : tl.constexpr,YB : tl.constexpr,ZB : tl.constexpr):
    xidx=tl.arange(0,XB)
    yidx=tl.arange(0,YB)
    zidx=tl.arange(0,ZB)

    idx=xidx[:,None,None]*YB*ZB+yidx[None,:,None]*ZB+zidx[None,None,:]

    X = tl.load(x_ptr+idx)

    xx,yy = tl.split(X)

    oidx=xidx[:,None]*YB+yidx[None,:]

   
    tl.store(output_ptr+oidx,xx)
    tl.store(output_ptr1+oidx,yy)



@pytest.mark.parametrize('para_type,data_type,XB,YB,ZB',
                         [
                             ['float32',torch.float32,8,8,2],
                             ['float16',torch.float16,8,8,2],
                             ['int8',torch.int8,8,128,2],
                             ['int8',torch.int8,8,8,2],
                         ]
                         )
def test_split(para_type,data_type,XB,YB,ZB):
    
    x = torch.randint(low=-128,high=128,size=(XB,YB,ZB),dtype=data_type).npu()
    
    a,b = torch.split(x,1,dim = -1)
    a = a.reshape(XB,YB)
    b = b.reshape(XB,YB)
    print(a)
    print(b)

    output = torch.randint(1, (XB,YB), dtype=data_type).npu()
    output1 = torch.randint(1, (XB,YB), dtype=data_type).npu()
    fn_npu_[1,1,1](output,x,output1, XB, YB, ZB, debug = True)

    print(output)
    print(output1)

    test_common.validate_cmp(para_type, a, output)
    test_common.validate_cmp(para_type, b, output1)


