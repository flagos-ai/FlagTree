# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 
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
    zidx=tl.arange(0,ZB)

    idx=xidx[:,None,None]*YB*ZB+yidx[None,:,None]*ZB+zidx[None,None,:]

    X = tl.load(x_ptr+idx)
    Y = tl.load(y_ptr+idx)

    ret = tl.interleave(X,Y)

    oidx=xidx[:,None,None]*YB*ZB*2+yidx[None,:,None]*ZB*2+tl.arange(0,2*ZB)[None,None,:]
   
    tl.store(output_ptr+oidx,ret) 



@pytest.mark.parametrize('para_type,data_type,XB,YB,ZB',
                         [
                             ['float32',torch.float32,2,64,16],
                             ['float32',torch.float32,8,8,4],
                             ['float16',torch.float16,2,64,16],
                             ['float16',torch.float16,8,8,4],
                             ['int8',torch.int8,2,64,32],
                             ['int8',torch.int8,8,8,4],
                         ]
                         )
def test_interleave(para_type,data_type,XB,YB,ZB):

    x = torch.full((XB,YB,ZB),100,dtype=data_type).npu()
    y = torch.full((XB,YB,ZB),30,dtype=data_type).npu()

    print(f"shape = {x.shape}")
    print(x.dtype)

    output = torch.randint(1, (XB,YB,ZB*2), dtype=data_type).npu()
    output1 = output
    print(f"output.dtype={output.dtype}")
    
    ans = torch.stack((x,y),dim=-1).reshape(XB,YB,ZB*2)
    print(ans)
    print(ans.shape)

    fn_npu_[1,1,1](output,x,y,XB,YB,ZB)
    print(output)

    test_common.validate_cmp(para_type, ans, output)