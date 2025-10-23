 # -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import pytest
import triton
import triton.language as tl
import time
import torch
import torch_npu
import test_common
from test_common import TestUtils

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
    mblk_idx = tl.arange(0, MNUMEL)
    nblk_idx = tl.arange(0, N) + tl.program_id(2) * N
    mmask = mblk_idx < MNUMEL
    nmask = nblk_idx < NNUMEL
    mask = (mmask[:, None]) & (nmask[None, :])
    idx = mblk_idx[:, None] * NNUMEL + nblk_idx[None, :]
    x = tl.load(in_ptr0 + idx, mask=mask, other=0)
    tmp1 = (x == cmp_val)
    tmp2 = tmp1.to(tl.float32)
    ret = tl.sum(tmp2, dim)
    tl.store(out_ptr0 + nblk_idx, ret, mask = nmask)

@triton.jit
def count_gt(in_ptr0, out_ptr0, cmp_val, dim : tl.constexpr, M : tl.constexpr, N : tl.constexpr, MNUMEL: tl.constexpr, NNUMEL: tl.constexpr):
    mblk_idx = tl.arange(0, MNUMEL)
    nblk_idx = tl.arange(0, N) + tl.program_id(2) * N
    mmask = mblk_idx < MNUMEL
    nmask = nblk_idx < NNUMEL
    mask = (mmask[:, None]) & (nmask[None, :])
    idx = mblk_idx[:, None] * NNUMEL + nblk_idx[None, :]
    x = tl.load(in_ptr0 + idx, mask=mask, other=0)
    tmp1 = (x > cmp_val)
    tmp2 = tmp1.to(tl.float32)
    ret = tl.sum(tmp2, dim)
    tl.store(out_ptr0 + nblk_idx, ret, mask = nmask)

@triton.jit
def count_lt(in_ptr0, out_ptr0, cmp_val, dim : tl.constexpr, M : tl.constexpr, N : tl.constexpr, MNUMEL: tl.constexpr, NNUMEL: tl.constexpr):
    mblk_idx = tl.arange(0, MNUMEL)
    nblk_idx = tl.arange(0, N) + tl.program_id(2) * N
    mmask = mblk_idx < MNUMEL
    nmask = nblk_idx < NNUMEL
    mask = (mmask[:, None]) & (nmask[None, :])
    idx = mblk_idx[:, None] * NNUMEL + nblk_idx[None, :]
    x = tl.load(in_ptr0 + idx, mask=mask, other=0)
    tmp1 = (x < cmp_val)
    tmp2 = tmp1.to(tl.float32)
    ret = tl.sum(tmp2, dim)
    tl.store(out_ptr0 + nblk_idx, ret, mask = nmask)

@pytest.mark.parametrize('shape', TestUtils.test_shape2d)
@pytest.mark.parametrize('dtype', ['int8'])
def test_count_dim0_common(shape, dtype):
    rblock = shape[1]
    xblock = shape[0]
    x0 = test_common.generate_tensor(shape, dtype).npu()

    if dtype == torch.int8:
        cmp_val = 8
    else:
        cmp_val = 0.5

    ans = standard_count(x0, cmp_val, 0, dtype)

    output = torch.zeros((shape[1],), dtype = torch.float32).npu()
    count[1, 1, rblock](x0, output, cmp_val, 0, xblock, 1, xblock, rblock)

    test_common.validate_cmp("float32", output, ans.to(torch.float32))

@pytest.mark.parametrize('shape', TestUtils.test_shape2d)
@pytest.mark.parametrize('dtype', ['float32', 'float16', 'int8'])
def test_count_gt_dim0_common(shape, dtype):
    rblock = shape[1]
    xblock = shape[0]
    x0 = test_common.generate_tensor(shape, dtype).npu()

    if dtype == torch.int8:
        cmp_val = 8
    else:
        cmp_val = 0.5

    ans = standard_count_gt(x0, cmp_val,0, dtype)

    output = torch.zeros((shape[1],), dtype = torch.float32).npu()
    count_gt[1, 1, rblock](x0, output, cmp_val, 0, xblock, 1, xblock, rblock)

    test_common.validate_cmp("float32", output, ans.to(torch.float32))

@pytest.mark.parametrize('shape', TestUtils.test_shape2d)
@pytest.mark.parametrize('dtype', ['float32', 'float16', 'int8'])
def test_count_lt_dim0_common(shape, dtype):
    rblock = shape[1]
    xblock = shape[0]
    x0 = test_common.generate_tensor(shape, dtype).npu()

    if dtype == torch.int8:
        cmp_val = 8
    else:
        cmp_val = 0.5

    ans = standard_count_lt(x0, cmp_val,0, dtype)

    output = torch.zeros((shape[1],), dtype = torch.float32).npu()
    count_lt[1, 1, rblock](x0, output, cmp_val, 0, xblock, 1, xblock, rblock)

    test_common.validate_cmp("float32", output, ans.to(torch.float32))
