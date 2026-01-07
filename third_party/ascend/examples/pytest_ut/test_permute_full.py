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
def fn_npu_021(output_ptr, x_ptr,XB : tl.constexpr,YB : tl.constexpr,ZB : tl.constexpr):
    xidx=tl.arange(0,XB)
    yidx=tl.arange(0,YB)
    zidx=tl.arange(0,ZB)
    idx=xidx[:,None,None]*YB*ZB+yidx[None,:,None]*ZB+zidx[None,None,:]
    
    # XB,YB,1
    X = tl.load(x_ptr + idx)

    ret=tl.permute(X,(0,2,1))

    oidx=xidx[:,None,None]*YB*ZB+zidx[None,:,None]*YB+yidx[None,None,:]

    tl.store(output_ptr + oidx, ret)

@triton.jit
def fn_npu_102(output_ptr, x_ptr,XB : tl.constexpr,YB : tl.constexpr,ZB : tl.constexpr):
    xidx=tl.arange(0,XB)
    yidx=tl.arange(0,YB)
    zidx=tl.arange(0,ZB)
    idx=xidx[:,None,None]*YB*ZB+yidx[None,:,None]*ZB+zidx[None,None,:]
    
    # XB,YB,1
    X = tl.load(x_ptr + idx)

    ret=tl.permute(X,(1,0,2))

    oidx=yidx[:,None,None]*XB*ZB+xidx[None,:,None]*ZB+zidx[None,None,:]

    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_210(output_ptr, x_ptr,XB : tl.constexpr,YB : tl.constexpr,ZB : tl.constexpr):
    xidx=tl.arange(0,XB)
    yidx=tl.arange(0,YB)
    zidx=tl.arange(0,ZB)
    idx=xidx[:,None,None]*YB*ZB+yidx[None,:,None]*ZB+zidx[None,None,:]
    
    # XB,YB,1
    X = tl.load(x_ptr + idx)

    ret=tl.permute(X,(2,1,0))

    oidx=zidx[:,None,None]*YB*XB+yidx[None,:,None]*XB+xidx[None,None,:]

    tl.store(output_ptr + oidx, ret)

@triton.jit
def fn_npu_201(output_ptr, x_ptr,XB : tl.constexpr,YB : tl.constexpr,ZB : tl.constexpr):
    xidx=tl.arange(0,XB)
    yidx=tl.arange(0,YB)
    zidx=tl.arange(0,ZB)
    idx=xidx[:,None,None]*YB*ZB+yidx[None,:,None]*ZB+zidx[None,None,:]
    
    # XB,YB,1
    X = tl.load(x_ptr + idx)

    ret=tl.permute(X,(2,0,1))

    oidx=zidx[:,None,None]*YB*XB+xidx[None,:,None]*YB+yidx[None,None,:]

    tl.store(output_ptr + oidx, ret)

@triton.jit
def fn_npu_120(output_ptr, x_ptr,XB : tl.constexpr,YB : tl.constexpr,ZB : tl.constexpr):
    xidx=tl.arange(0,XB)
    yidx=tl.arange(0,YB)
    zidx=tl.arange(0,ZB)
    idx=xidx[:,None,None]*YB*ZB+yidx[None,:,None]*ZB+zidx[None,None,:]
    
    # XB,YB,1
    X = tl.load(x_ptr + idx)

    ret=tl.permute(X,(1,2,0))

    oidx=yidx[:,None,None]*ZB*XB+zidx[None,:,None]*XB+xidx[None,None,:]

    tl.store(output_ptr + oidx, ret)


@pytest.mark.parametrize('para_type,data_type,XB,YB,ZB',
                         [
                            # ['float32',eval('torch.float32'),2,4,3],
                            ['float32',eval('torch.float32'),2,4,8],
                            # ['float32',eval('torch.float32'),2,4,37],
                            ['float32',eval('torch.float32'),2,4,64],
                            # ['float32',eval('torch.float32'),2,4,781],

                            # ['float16',eval('torch.float16'),2,4,3],
                            ['float16',eval('torch.float16'),2,4,8],
                            # ['float16',eval('torch.float16'),2,4,37],
                            ['float16',eval('torch.float16'),2,4,64],
                            # ['float16',eval('torch.float16'),2,4,781],

                            # ['int8',eval('torch.int8'),2,4,3],
                            ['int8',eval('torch.int8'),2,4,8],
                            # ['int8',eval('torch.int8'),2,4,37],
                            ['int8',eval('torch.int8'),2,4,64],
                            # ['int8',eval('torch.int8'),2,4,781],
                         ]
                         )
def test_permute(para_type,data_type,XB,YB,ZB):

    x = torch.randint(low=0,high=2,size=(XB,YB,ZB),dtype=data_type).npu()
    
    output = torch.randint(1, (XB,ZB,YB), dtype=data_type).npu()
    torch_021 = torch.permute(x,(0,2,1))
    fn_npu_021[1,1,1](output, x, XB, YB, ZB)
    torch.testing.assert_close(output,torch_021)

    print(" test permute 021 passed")

    output = torch.randint(1, (YB,XB,ZB), dtype=data_type).npu()
    torch_102 = torch.permute(x,(1,0,2))
    fn_npu_102[1,1,1](output, x, XB, YB, ZB)
    torch.testing.assert_close(output,torch_102)

    print(" test permute 102 passed")

    output = torch.randint(1, (ZB,XB,YB), dtype=data_type).npu()
    torch_201 = torch.permute(x, (2,0,1))
    fn_npu_201[1,1,1](output, x, XB, YB, ZB)
    torch.testing.assert_close(output, torch_201)

    print(" test permute 201 passed")

    output = torch.randint(1, (ZB,YB,XB), dtype=data_type).npu()
    torch_210 = torch.permute(x, (2,1,0))
    fn_npu_210[1,1,1](output, x, XB, YB, ZB)
    torch.testing.assert_close(output, torch_210)

    print(" test permute 210 passed")

    output = torch.randint(1, (YB,ZB,XB), dtype=data_type).npu()
    torch_120 = torch.permute(x, (1,2,0))
    fn_npu_120[1,1,1](output, x, XB, YB, ZB)
    torch.testing.assert_close(output, torch_120)

    print(" test permute 120 passed")