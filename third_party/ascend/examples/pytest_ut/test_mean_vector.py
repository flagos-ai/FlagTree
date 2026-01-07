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


import pytest

import triton
import triton.language as tl
import time

import torch
import torch_npu
import test_common

def standard_mean(x0,dim,dtype):
    res = torch.mean(x0, dim,keepdim = True,dtype = dtype)
    return res

@triton.jit
def triton_mean_dim0(in_ptr0, out_ptr0, N : tl.constexpr, NUMEL: tl.constexpr):
    idx_block = tl.arange(0,NUMEL)

    x=tl.load(in_ptr0+idx_block, mask = idx_block<N,other = 0)

    if x.dtype == tl.bfloat16:
        ret = (tl.sum(x.to(tl.float32),0)/N).to(tl.bfloat16)
    elif x.dtype == tl.float16:
        ret = tl.sum(x,0)/N
    else :
        ret = tl.sum(x.to(tl.float32),0)/N

    tl.store(out_ptr0+idx_block, ret, mask = idx_block<1)

types=[
    (torch.float32,'float32'),
    # (torch.float16,'float16'), TODO: should fix reduceConverter's bug
    # (torch.bfloat16,'bfloat16'),  TODO: waiting for supporting or testing
    (torch.int8,'int8'),
    # (torch.int16,'int16'),  TODO: waiting for supporting or testing
    # (torch.int32,'int32'),  TODO: waiting for supporting or testing
    # (torch.int64,'int64'),  TODO: waiting for supporting or testing
]

# if shape axis = 32/256 , then actual shape = axis/element_size()
shapes=[
    (3,32),
    (-32,32),
    (37,64),
    (-256,256),
    (781,1024),
]

map_for_64_t = {37:(31,32)}

@pytest.mark.parametrize('dtype, sigtype',types)
@pytest.mark.parametrize('N, NUMEL',shapes)
def test_mean_dim0(dtype, sigtype, N, NUMEL):

    N = (-N)//torch.tensor(0,dtype=dtype).element_size() if N<0 else N

    if sigtype == 'int64':
        N = map_for_64_t[N][0] if N in map_for_64_t else N
        NUMEL = map_for_64_t[N][1] if N in map_for_64_t else NUMEL
    
    res_dtype = dtype
    res_sigtype = sigtype
    should_cast_to_fp32 = ['int8','int16','int32','int64','float32']

    if sigtype in should_cast_to_fp32:
        res_dtype = torch.float32
        res_sigtype = 'float32'

    print(f"sum : ({N},) {dtype} {sigtype}")
    x0 = test_common.generate_tensor(shape = (N,),dtype = sigtype)

    ans = standard_mean(x0, 0, res_dtype)

    x0=x0.npu()
    print(ans)
   
    output = torch.zeros((1,), dtype = res_dtype).npu()
    triton_mean_dim0[1,1,1](x0, output, N = N, NUMEL = NUMEL, debug = True)
    print(output)

    test_common.validate_cmp(res_sigtype,output,ans)