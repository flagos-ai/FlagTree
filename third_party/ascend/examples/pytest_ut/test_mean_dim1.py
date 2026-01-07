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
    res = torch.mean(x0, dim,dtype = dtype)
    return res

@triton.jit
def triton_mean_dim1(in_ptr0, out_ptr0, M : tl.constexpr, N : tl.constexpr, MNUMEL: tl.constexpr, NNUMEL: tl.constexpr):
    mblk_idx = tl.arange(0,MNUMEL)
    nblk_idx = tl.arange(0,NNUMEL)

    mmask = mblk_idx<M
    nmask = nblk_idx<N

    mask = (mmask[:,None])&(nmask[None,:])

    idx = mblk_idx[:,None]*N+nblk_idx[None,:]

    x=tl.load(in_ptr0+idx, mask = mask, other = 0)

    if x.dtype == tl.bfloat16:
        ret = (tl.sum(x.to(tl.float32),1)/N).to(tl.bfloat16)
    elif x.dtype == tl.float16:
        ret = tl.sum(x,1)/N
    else :
        ret = tl.sum(x.to(tl.float32),1)/N

    tl.store(out_ptr0+mblk_idx, ret, mask = mmask)

types=[
    (torch.float32,'float32'),
    (torch.float16,'float16'),
    # (torch.bfloat16,'bfloat16'),  TODO: waiting for supporting or testing
    (torch.int8,'int8'),
    # (torch.int16,'int16'),  TODO: waiting for supporting or testing
    # (torch.int32,'int32'),  TODO: waiting for supporting or testing
    # (torch.int64,'int64'),  TODO: waiting for supporting or testing
]

# if shape axis = 32/256 , then actual shape = axis/element_size()
shapes=[
    (57,3,64,16), (57,-32,64,32), (57,37,64,64), (57,-256,64,256), (57,263,64,512),
    (64,3,64,16), (64,-32,64,32), (64,37,64,64), (64,-256,64,256), (64,263,64,512),
    (3,3,8,8), (-32,3,32,8), (37,3,64,8), (-256,3,256,8), (263,3,512,8),
    (3,1,8,8), (-32,1,32,8), (37,1,64,8), (-256,1,256,8), (263,1,512,8),
]

map_for_64_t = {37:(31,32)}
map_for_32_t = {263:(137,256)}

# @pytest.mark.parametrize('dtype, sigtype',[(torch.float32,'float32'),])
@pytest.mark.parametrize('M, N, MNUMEL, NNUMEL',[(57,3,64,16),(64,-32,64,32),(37,3,64,8),(263,1,512,8),(-256,3,256,8),])
@pytest.mark.parametrize('dtype, sigtype',types)
# @pytest.mark.parametrize('M, N, MNUMEL, NNUMEL',shapes)
def test_mean_dim1(dtype, sigtype, M, N, MNUMEL, NNUMEL):

    M = (-M)//torch.tensor(0,dtype=dtype).element_size() if M<0 else M
    N = (-N)//torch.tensor(0,dtype=dtype).element_size() if N<0 else N

    if sigtype == 'int64':
        M = map_for_64_t[M][0] if M in map_for_64_t else M
        MNUMEL = map_for_64_t[M][1] if M in map_for_64_t else MNUMEL
        N = map_for_64_t[N][0] if N in map_for_64_t else N
        NNUMEL = map_for_64_t[N][1] if N in map_for_64_t else NNUMEL
    
    res_dtype = dtype
    res_sigtype = sigtype
    should_cast_to_fp32 = ['int8','int16','int32','int64','float32','bfloat16']

    if sigtype in should_cast_to_fp32:
        M = map_for_32_t[M][0] if M in map_for_32_t else M
        MNUMEL = map_for_32_t[M][1] if M in map_for_32_t else MNUMEL
        N = map_for_32_t[N][0] if N in map_for_32_t else N
        NNUMEL = map_for_32_t[N][1] if N in map_for_32_t else NNUMEL
        if sigtype != 'bfloat16':
            res_dtype = torch.float32
            res_sigtype = 'float32'

    print(f"sum : ({M}, {N}) {dtype} {sigtype}")
    x0 = test_common.generate_tensor(shape = (M,N),dtype = sigtype)

    ans = standard_mean(x0, 1, res_dtype)

    x0=x0.npu()
    print(ans)
   
    output = torch.zeros((M,), dtype = res_dtype).npu()
    triton_mean_dim1[1,1,1](x0, output, M = M, N = N,MNUMEL = MNUMEL, NNUMEL = NNUMEL, debug = True)
    print(output)

    test_common.validate_cmp(res_sigtype,output,ans)