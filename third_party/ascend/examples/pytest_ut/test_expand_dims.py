# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 
import triton
import triton.language as tl

import torch
import torch_npu
import pytest

@triton.jit
def fn_npu_(output_ptr, x_ptr,XB : tl.constexpr,YB : tl.constexpr,ZB : tl.constexpr):
    xidx=tl.arange(0,XB)
    yidx=tl.arange(0,YB)
    zidx=tl.arange(0,ZB)

    idx=xidx[:,None,None]*YB*ZB+yidx[None,:,None]*ZB+zidx[None,None,:]

    X = tl.load(x_ptr+idx)

    ret = tl.expand_dims(X,2)

    oidx=xidx[:,None,None,None]*YB*ZB+yidx[None,:,None,None]*ZB+tl.arange(0,1)[None,None,:,None]+zidx[None,None,None,:]

    tl.store(output_ptr+oidx,ret)

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
    a = x.unsqueeze(2)

    print(f"shape = {x.shape}")
    print(x.dtype)
    print(a[0,0:16,0,0])

    output = torch.randint(1, (XB,YB,1,ZB), dtype=data_type).npu()

    print(f"output.dtype={output.dtype}")

    fn_npu_[1,1,1](output,x, XB=XB, YB=YB, ZB=ZB, debug=True)
    print(output[0,0:16,0,0])

    torch.testing.assert_close(output,a)