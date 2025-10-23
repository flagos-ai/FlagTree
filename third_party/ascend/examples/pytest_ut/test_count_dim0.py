 # -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import pytest
import triton
import triton.language as tl
import time
import torch
import torch_npu
import test_common

def standard_count(x0, cmp_val, dim, dtype):
    res = (x0 == cmp_val).sum(dim=dim)
    return res

def standard_count_gt(x0, cmp_val, dim, dtype):
    res = (x0 > cmp_val).sum(dim=dim)
    return res

def standard_count_lt(x0, cmp_val, dim, dtype):
    res = (x0 < cmp_val).sum(dim=dim)
    return res

@triton.jit
def count(in_ptr0, out_ptr0, cmp_val, dim : tl.constexpr, M : tl.constexpr, N : tl.constexpr, MNUMEL: tl.constexpr, NNUMEL: tl.constexpr):
    mblk_idx = tl.arange(0,MNUMEL)
    nblk_idx = tl.arange(0,NNUMEL)
    mmask = mblk_idx < M
    nmask = nblk_idx < N
    mask = (mmask[:,None]) & (nmask[None,:])
    idx = mblk_idx[:,None]*N + nblk_idx[None,:]
    x = tl.load(in_ptr0+idx, mask = mask, other = 0)
    tmp1 = (x == cmp_val)
    tmp2 = tmp1.to(tl.float32)
    ret = tl.sum(tmp2, dim)
    tl.store(out_ptr0 + nblk_idx, ret, mask = nmask)

@triton.jit
def count_gt(in_ptr0, out_ptr0, cmp_val, dim : tl.constexpr, M : tl.constexpr, N : tl.constexpr, MNUMEL: tl.constexpr, NNUMEL: tl.constexpr):
    mblk_idx = tl.arange(0,MNUMEL)
    nblk_idx = tl.arange(0,NNUMEL)
    mmask = mblk_idx < M
    nmask = nblk_idx < N
    mask = (mmask[:,None]) & (nmask[None,:])
    idx = mblk_idx[:,None]*N + nblk_idx[None,:]
    x = tl.load(in_ptr0+idx, mask = mask, other = 0)
    tmp1 = (x > cmp_val)
    tmp2 = tmp1.to(tl.float32)
    ret = tl.sum(tmp2, dim)
    tl.store(out_ptr0 + nblk_idx, ret, mask = nmask)

@triton.jit
def count_lt(in_ptr0, out_ptr0, cmp_val, dim : tl.constexpr, M : tl.constexpr, N : tl.constexpr, MNUMEL: tl.constexpr, NNUMEL: tl.constexpr):
    mblk_idx = tl.arange(0,MNUMEL)
    nblk_idx = tl.arange(0,NNUMEL)
    mmask = mblk_idx < M
    nmask = nblk_idx < N
    mask = (mmask[:,None]) & (nmask[None,:])
    idx = mblk_idx[:,None]*N + nblk_idx[None,:]
    x = tl.load(in_ptr0+idx, mask = mask, other = 0)
    tmp1 = (x < cmp_val)
    tmp2 = tmp1.to(tl.float32)
    ret = tl.sum(tmp2, dim)
    tl.store(out_ptr0 + nblk_idx, ret, mask = nmask)


shapes=[
    (57,3,64,16), (57,-32,64,32), (57,37,64,64),
    (64,3,64,16), (64,-32,64,32), (64,37,64,64),
    (3,3,8,8), (-32,3,32,8), (37,3,64,8),
    (3,1,8,8), (-32,1,32,8), (37,1,64,8)
]

map_for_64_t = {37:(31,32),263:(107,128)}
map_for_32_t = {263:(137,256)}


types0 = [
    (torch.int8,'int8'),
]
@pytest.mark.parametrize('dtype, sigtype',types0)
@pytest.mark.parametrize('M, N, MNUMEL, NNUMEL',shapes)
def test_count_eq_dim0_common(dtype, sigtype, M, N, MNUMEL, NNUMEL):
    M = (-M)//torch.tensor(0,dtype=dtype).element_size() if M<0 else M
    N = (-N)//torch.tensor(0,dtype=dtype).element_size() if N<0 else N

    if sigtype == 'int64':
        M = map_for_64_t[M][0] if M in map_for_64_t else M
        MNUMEL = map_for_64_t[M][1] if M in map_for_64_t else MNUMEL
        N = map_for_64_t[N][0] if N in map_for_64_t else N
        NNUMEL = map_for_64_t[N][1] if N in map_for_64_t else NNUMEL

    elif sigtype == 'float32' or sigtype == 'bfloat16' or sigtype == 'int32':
        M = map_for_32_t[M][0] if M in map_for_32_t else M
        MNUMEL = map_for_32_t[M][1] if M in map_for_32_t else MNUMEL
        N = map_for_32_t[N][0] if N in map_for_32_t else N
        NNUMEL = map_for_32_t[N][1] if N in map_for_32_t else NNUMEL

    print(f"sum : ({M}, {N}) {dtype} {sigtype}")
    cmp_val = 8
    x0 = test_common.generate_tensor(shape = (M,N),dtype = sigtype)
    ans = standard_count(x0, cmp_val,0, dtype)
    x0 = x0.npu()
    print(ans)
    output = torch.zeros((N,), dtype = torch.float32).npu()
    count[1,1,1](x0, output, cmp_val, 0, M = M, N = N,MNUMEL = MNUMEL, NNUMEL = NNUMEL, debug = True)
    print(output)
    test_common.validate_cmp('float32', output, ans.to(torch.float32))

#-------------------------------------------------------------------------------------

types1 = [
    (torch.float32,'float32'),
    (torch.float32,'float16'),
    (torch.int8,'int8'),
]
@pytest.mark.parametrize('dtype, sigtype',types1)
@pytest.mark.parametrize('M, N, MNUMEL, NNUMEL',shapes)
def test_count_gt_dim0_common(dtype, sigtype, M, N, MNUMEL, NNUMEL):
    M = (-M)//torch.tensor(0,dtype=dtype).element_size() if M<0 else M
    N = (-N)//torch.tensor(0,dtype=dtype).element_size() if N<0 else N

    if sigtype == 'int64':
        M = map_for_64_t[M][0] if M in map_for_64_t else M
        MNUMEL = map_for_64_t[M][1] if M in map_for_64_t else MNUMEL
        N = map_for_64_t[N][0] if N in map_for_64_t else N
        NNUMEL = map_for_64_t[N][1] if N in map_for_64_t else NNUMEL

    elif sigtype == 'float32' or sigtype == 'bfloat16' or sigtype == 'int32':
        M = map_for_32_t[M][0] if M in map_for_32_t else M
        MNUMEL = map_for_32_t[M][1] if M in map_for_32_t else MNUMEL
        N = map_for_32_t[N][0] if N in map_for_32_t else N
        NNUMEL = map_for_32_t[N][1] if N in map_for_32_t else NNUMEL

    print(f"sum : ({M}, {N}) {dtype} {sigtype}")
    if dtype == torch.int8:
        cmp_val = 8
    else:
        cmp_val = 0.5
    x0 = test_common.generate_tensor(shape = (M,N),dtype = sigtype)
    ans = standard_count_gt(x0, cmp_val,0, dtype)
    x0 = x0.npu()
    print(ans)
    output = torch.zeros((N,), dtype = torch.float32).npu()
    count_gt[1,1,1](x0, output, cmp_val, 0, M = M, N = N,MNUMEL = MNUMEL, NNUMEL = NNUMEL, debug = True)
    print(output)
    test_common.validate_cmp("float32", output, ans.to(torch.float32))


shapes1=[
    (64,3,64,16), (64,-32,64,32), (64,37,64,64)
]
@pytest.mark.parametrize('dtype, sigtype',types1)
@pytest.mark.parametrize('M, N, MNUMEL, NNUMEL',shapes1)
def test_count_lt_dim0_common(dtype, sigtype, M, N, MNUMEL, NNUMEL):
    M = (-M)//torch.tensor(0,dtype=dtype).element_size() if M<0 else M
    N = (-N)//torch.tensor(0,dtype=dtype).element_size() if N<0 else N

    if sigtype == 'int64':
        M = map_for_64_t[M][0] if M in map_for_64_t else M
        MNUMEL = map_for_64_t[M][1] if M in map_for_64_t else MNUMEL
        N = map_for_64_t[N][0] if N in map_for_64_t else N
        NNUMEL = map_for_64_t[N][1] if N in map_for_64_t else NNUMEL

    elif sigtype == 'float32' or sigtype == 'bfloat16' or sigtype == 'int32':
        M = map_for_32_t[M][0] if M in map_for_32_t else M
        MNUMEL = map_for_32_t[M][1] if M in map_for_32_t else MNUMEL
        N = map_for_32_t[N][0] if N in map_for_32_t else N
        NNUMEL = map_for_32_t[N][1] if N in map_for_32_t else NNUMEL

    print(f"sum : ({M}, {N}) {dtype} {sigtype}")
    if dtype == torch.int8:
        cmp_val = 8
    else:
        cmp_val = 0.5
    x0 = test_common.generate_tensor(shape = (M,N),dtype = sigtype)
    ans = standard_count_lt(x0, cmp_val,0, dtype)
    x0 = x0.npu()
    print(ans)
    output = torch.zeros((N,), dtype = torch.float32).npu()
    count_lt[1,1,1](x0, output, cmp_val, 0, M = M, N = N,MNUMEL = MNUMEL, NNUMEL = NNUMEL, debug = True)
    print(output)
    test_common.validate_cmp("float32", output, ans.to(torch.float32))
