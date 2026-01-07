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
def count(in_ptr0, out_ptr0, cmp_val, dim: tl.constexpr, M: tl.constexpr, N: tl.constexpr, MNUMEL: tl.constexpr,
          NNUMEL: tl.constexpr):
    mblk_idx = tl.arange(0, M) + tl.program_id(1) * M
    nblk_idx = tl.arange(0, NNUMEL)
    mmask = mblk_idx < MNUMEL
    nmask = nblk_idx < NNUMEL
    mask = (mmask[:, None]) & (nmask[None, :])
    idx = mblk_idx[:, None] * NNUMEL + nblk_idx[None, :]
    x = tl.load(in_ptr0 + idx, mask=mask, other=0)
    tmp1 = (x == cmp_val)
    tmp2 = tmp1.to(tl.float32)
    ret = tl.sum(tmp2, dim)
    tl.store(out_ptr0 + mblk_idx, ret, mask=mmask)


@triton.jit
def count_gt(in_ptr0, out_ptr0, cmp_val, dim: tl.constexpr, M: tl.constexpr, N: tl.constexpr, MNUMEL: tl.constexpr,
             NNUMEL: tl.constexpr):
    mblk_idx = tl.arange(0, M) + tl.program_id(1) * M
    nblk_idx = tl.arange(0, NNUMEL)
    mmask = mblk_idx < MNUMEL
    nmask = nblk_idx < NNUMEL
    mask = (mmask[:, None]) & (nmask[None, :])
    idx = mblk_idx[:, None] * NNUMEL + nblk_idx[None, :]
    x = tl.load(in_ptr0 + idx, mask=mask, other=0)
    tmp1 = (x > cmp_val)
    tmp2 = tmp1.to(tl.float32)
    ret = tl.sum(tmp2, dim)
    tl.store(out_ptr0 + mblk_idx, ret, mask=mmask)


@triton.jit
def count_lt(in_ptr0, out_ptr0, cmp_val, dim: tl.constexpr, M: tl.constexpr, N: tl.constexpr, MNUMEL: tl.constexpr,
             NNUMEL: tl.constexpr):
    mblk_idx = tl.arange(0, M) + tl.program_id(1) * M
    nblk_idx = tl.arange(0, NNUMEL)
    mmask = mblk_idx < MNUMEL
    nmask = nblk_idx < NNUMEL
    mask = (mmask[:, None]) & (nmask[None, :])
    idx = mblk_idx[:, None] * NNUMEL + nblk_idx[None, :]
    x = tl.load(in_ptr0 + idx, mask=mask, other=0)
    tmp1 = (x < cmp_val)
    tmp2 = tmp1.to(tl.float32)
    ret = tl.sum(tmp2, dim)
    tl.store(out_ptr0 + mblk_idx, ret, mask=mmask)

@pytest.mark.parametrize('shape', TestUtils.test_shape2d)
@pytest.mark.parametrize('dtype', ['int8'])
def test_count_dim1_common(shape, dtype):
    rblock = shape[1]
    xblock = shape[0]
    x0 = test_common.generate_tensor(shape, dtype).npu()

    if dtype == torch.int8:
        cmp_val = 8
    else:
        cmp_val = 0.5

    ans = standard_count(x0, cmp_val,1, dtype)

    output = torch.zeros((shape[0],), dtype = torch.float32).npu()
    count[1, xblock, 1](x0, output, cmp_val, 1, 1, rblock, xblock, rblock)

    test_common.validate_cmp("float32", output, ans.to(torch.float32))

@pytest.mark.parametrize('shape', TestUtils.test_shape2d)
@pytest.mark.parametrize('dtype', ['float32', 'float16', 'int8'])
def test_count_gt_dim1_common(shape, dtype):
    rblock = shape[1]
    xblock = shape[0]
    x0 = test_common.generate_tensor(shape, dtype).npu()

    if dtype == torch.int8:
        cmp_val = 8
    else:
        cmp_val = 0.5

    ans = standard_count_gt(x0, cmp_val,1, dtype)

    output = torch.zeros((shape[0],), dtype = torch.float32).npu()
    count_gt[1, xblock, 1](x0, output, cmp_val, 1, 1, rblock, xblock, rblock)

    test_common.validate_cmp("float32", output, ans.to(torch.float32))

@pytest.mark.parametrize('shape', TestUtils.test_shape2d)
@pytest.mark.parametrize('dtype', ['float32', 'float16', 'int8'])
def test_count_lt_dim1_common(shape, dtype):
    rblock = shape[1]
    xblock = shape[0]
    x0 = test_common.generate_tensor(shape, dtype).npu()

    if dtype == torch.int8:
        cmp_val = 8
    else:
        cmp_val = 0.5

    ans = standard_count_lt(x0, cmp_val,1, dtype)

    output = torch.zeros((shape[0],), dtype = torch.float32).npu()
    count_lt[1, xblock, 1](x0, output, cmp_val, 1, 1, rblock, xblock, rblock)

    test_common.validate_cmp("float32", output, ans.to(torch.float32))