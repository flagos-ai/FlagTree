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

@triton.jit
def fn_npu_(output_ptr, x_ptr,y_ptr,z_ptr,output_ptr1,XB : tl.constexpr,YB : tl.constexpr,ZB : tl.constexpr):
    xidx=tl.arange(0,XB)
    yidx=tl.arange(0,YB)
    zidx=tl.arange(0,ZB)
    idx=xidx[:,None,None]*YB*ZB+yidx[None,:,None]*ZB+zidx[None,None,:]
    block_ptr_in=tl.make_block_ptr(
        base = x_ptr,
        shape = (XB,YB,ZB),
        strides = (YB*ZB,ZB,1),
        offsets = (0,0,0),
        block_shape = (XB,YB,ZB),
        order = (2,1,0),
    )
    X = tl.load(block_ptr_in)

    oidx=xidx[:,None,None]*YB*ZB+yidx[None,:,None]*ZB+zidx[None,None,:]

    block_ptr_out=tl.make_block_ptr(
        base = output_ptr,
        shape = (XB,YB,ZB),
        strides = (YB*ZB,ZB,1),
        offsets = (0,0,0),
        block_shape = (XB,YB,ZB),
        order = (2,1,0),
    )
    tl.store(block_ptr_out,X)

paras = [
    ('*fp32',eval('torch.float32'),2,256,16),
    ('*fp32',eval('torch.float32'),8,8,4),
    ('*fp16',eval('torch.float16'),2,256,16),
    ('*fp16',eval('torch.float16'),8,8,4),
    ('*i8',eval('torch.int8'),2,256,16),
    ('*i8',eval('torch.int8'),8,8,4),
]

@pytest.mark.parametrize('para_type,data_type,XB,YB,ZB', paras)
def test_npu(para_type,data_type,XB,YB,ZB):

    x = torch.randint(low=-128,high=128,size=(XB,YB,ZB),dtype=data_type).npu()
    y = torch.randint(low=-128,high=128,size=(XB,YB,ZB),dtype=data_type).npu()
    z = torch.randint(low=-128,high=128,size=(XB,YB,ZB),dtype=data_type).npu()

    print(f"shape = {x.shape}")
    print(x.dtype)

    output = torch.randint(1, (XB,YB,ZB), dtype=data_type).npu()
    output1 = output
    print(f"output.dtype={output.dtype}")
    
    a = x
    print(a)
    fn_npu_[1,1,1](output,x,y,z,output1, XB=XB, YB=YB, ZB=ZB, debug=True)
    print(output)
    torch.testing.assert_close(output,a)
