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
def fn_npu_(output_ptr, x_ptr,y_ptr,XB : tl.constexpr,YB : tl.constexpr,ZB : tl.constexpr):
    xidx=tl.arange(0,XB)
    yidx=tl.arange(0,YB)

    idx=xidx[:,None]*YB+yidx[None,:]

    X = tl.load(x_ptr+idx)
    Y = tl.load(y_ptr+idx)

    ret = tl.join(X,Y)

    oidx=xidx[:,None,None]*YB*2+yidx[None,:,None]*2+tl.arange(0,2)[None,None,:]
   
    tl.store(output_ptr+oidx,ret)

@pytest.mark.parametrize('para_type,data_type,XB,YB,ZB',
                         [
                             ['float32',torch.float32,4,64,4],
                             ['float32',torch.float32,8,8,4],
                             ['float16',torch.float16,4,64,4],
                             ['float16',torch.float16,8,8,4],
                             ['int8',torch.int8,4,128,4],
                             ['int8',torch.int8,8,8,4],
                         ]
                         )
def test_join(para_type,data_type,XB,YB,ZB):
    x = torch.full((XB,YB),100,dtype=data_type).npu()
    y = torch.full((XB,YB),30,dtype=data_type).npu()

    ans = torch.stack((x,y),dim=-1)
    print(ans)

    output = torch.randint(1, (XB,YB,2), dtype=data_type).npu()
    fn_npu_[1,1,1](output,x,y,XB, YB, ZB, debug = True)

    print(output)
    test_common.validate_cmp(para_type, ans, output)
