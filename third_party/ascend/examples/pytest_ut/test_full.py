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
import test_common

import torch
import torch_npu
import pytest

@triton.jit
def fn_npu_f32(output_ptr,XB : tl.constexpr,YB : tl.constexpr,ZB : tl.constexpr):
    xidx=tl.arange(0,XB)
    yidx=tl.arange(0,YB)
    zidx=tl.arange(0,ZB)

    ret = tl.full((XB,YB,ZB),value = 100,dtype = tl.float32)

    oidx=xidx[:,None,None]*YB*ZB+yidx[None,:,None]*ZB+zidx[None,None,:]

    tl.store(output_ptr+oidx,ret)

@triton.jit
def fn_npu_f16(output_ptr,XB : tl.constexpr,YB : tl.constexpr,ZB : tl.constexpr):
    xidx=tl.arange(0,XB)
    yidx=tl.arange(0,YB)
    zidx=tl.arange(0,ZB)

    ret = tl.full((XB,YB,ZB),value = 100,dtype = tl.float16)

    oidx=xidx[:,None,None]*YB*ZB+yidx[None,:,None]*ZB+zidx[None,None,:]

    tl.store(output_ptr+oidx,ret)

@triton.jit
def fn_npu_i8(output_ptr,XB : tl.constexpr,YB : tl.constexpr,ZB : tl.constexpr):
    xidx=tl.arange(0,XB)
    yidx=tl.arange(0,YB)
    zidx=tl.arange(0,ZB)

    ret = tl.full((XB,YB,ZB),value = 100,dtype = tl.int8)

    oidx=xidx[:,None,None]*YB*ZB+yidx[None,:,None]*ZB+zidx[None,None,:]

    tl.store(output_ptr+oidx,ret)

testlist = [
    (fn_npu_f32,'float32',torch.float32,2,256,16),
    (fn_npu_f32,'float32',torch.float32,8,8,4),

    (fn_npu_f16,'float16',torch.float16,2,256,16),
    (fn_npu_f16,'float16',torch.float16,8,8,4),

    (fn_npu_i8,'int8',torch.int8,2,256,16),
    (fn_npu_i8,'int8',torch.int8,8,8,4),
]

@pytest.mark.parametrize('testfunc, sigtype, dtype, XB, YB, ZB',testlist)
def test_npu(testfunc, sigtype, dtype, XB, YB, ZB):
    
    x = torch.full((XB,YB,ZB),100,dtype=dtype).npu()

    print(f"shape = {x.shape}")
    print(x.dtype)
    print(x[0,0:16,0])

    output = torch.randint(1, (XB,YB,ZB), dtype=dtype).npu()

    print(f"output.dtype={output.dtype}")

    testfunc[1,1,1](output,XB,YB,ZB,debug=True)
    print(output[0,0:16,0])

    test_common.validate_cmp(sigtype,output,x)
