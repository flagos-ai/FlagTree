# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import time

import pytest
import torch
import torch_npu

import triton
import triton.language as tl
import test_common


@triton.jit
def promote_to_tensor(x):
    # Addition promotes to tensor for us
    return x + tl.zeros((1, ), tl.int1)


@triton.jit
def minimum_with_index(a_value, a_index, b_value, b_index):
    mask = a_value < b_value
    equal = a_value == b_value
    if promote_to_tensor(a_value).dtype.is_floating():
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan and not b_isnan
        # Consider NaNs as equal
        equal |= a_isnan and b_isnan
    # Prefer lowest index if values are equal
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@triton.jit
def triton_min_5d_dim4_keepdim(in_ptr0, in_ptr1, out_ptr0, out_ptr1, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
                               K: tl.constexpr, Z: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    kblk_idx = tl.arange(0, K)
    zblk_idx = tl.arange(0, Z)
    idx = lblk_idx[:, None, None, None, None] * Z * K * N * M + mblk_idx[None, :, None, None, None] * Z * K * N + \
          nblk_idx[None, None, :, None, None] * Z * K + kblk_idx[None, None, None, :, None] * Z + zblk_idx[None, None, None, None, :]
    x = tl.load(in_ptr0 + idx)
    x1 = tl.load(in_ptr1 + idx)
    ret, ret1 = tl.reduce((x, x1), 4, minimum_with_index, keep_dims=True)
    zblk_idx = tl.arange(0, 1)
    odx = lblk_idx[:, None, None, None, None] * K * N * M + mblk_idx[None, :, None, None, None] * K * N + \
          nblk_idx[None, None, :, None, None] * K + kblk_idx[None, None, None, :, None]  \
          + zblk_idx[None, None, None, None, :]
    tl.store(out_ptr0 + odx, ret)
    tl.store(out_ptr1 + odx, ret1)


testlist = [
    # 5D
    (triton_min_5d_dim4_keepdim, (1, 1, 1, 1, 1)),
    (triton_min_5d_dim4_keepdim, (2, 2, 2, 2, 2)),
    (triton_min_5d_dim4_keepdim, (9, 3, 2, 4, 17)),
    (triton_min_5d_dim4_keepdim, (3, 11, 1, 3, 42)),
    (triton_min_5d_dim4_keepdim, (2, 51, 3, 13, 1)),
    (triton_min_5d_dim4_keepdim, (129, 1, 5, 1, 4)),
    (triton_min_5d_dim4_keepdim, (203, 1, 2, 2, 3)),
    (triton_min_5d_dim4_keepdim, (512, 1, 1, 1, 1)),
    (triton_min_5d_dim4_keepdim, (3, 1, 1, 2, 600)),
    (triton_min_5d_dim4_keepdim, (1, 1, 1, 1, 1024)),
    (triton_min_5d_dim4_keepdim, (15, 2, 2, 2, 54)),
    (triton_min_5d_dim4_keepdim, (2, 91, 4, 2, 4)),
    (triton_min_5d_dim4_keepdim, (1, 1, 3, 2, 600)),
    (triton_min_5d_dim4_keepdim, (5, 2, 4, 1, 26)),
    (triton_min_5d_dim4_keepdim, (2, 2, 2, 4, 8)),
]

typelist = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'bool']

ids = ["{}-{}".format(testfunc.__name__, "-".join(map(str, shape))) for testfunc, shape in testlist]


@pytest.mark.parametrize('testfunc, shape', testlist, ids=ids)
@pytest.mark.parametrize('sigtype', typelist)
def test_min_dim4_keepdim(testfunc, sigtype, shape):
    dtype = eval('torch.' + sigtype)
    x0 = torch.randn(shape).to(dtype).npu()

    x1 = torch.arange(x0.numel()).view(x0.shape).npu().to(torch.int32)
    if 'int' in sigtype:
        ans, ans1 = torch.min(x0.to(torch.int64), 4)
        ans = ans.to(dtype)
    else:
        ans, ans1 = torch.min(x0, 4)
    output = torch.zeros(shape[0:4], dtype=dtype).npu()
    output1 = torch.zeros(shape[0:4], dtype=torch.int32).npu()
    testfunc[(1, )](x0, x1, output, output1, *shape, debug=True)
    test_common.validate_cmp(sigtype, output, ans)
    test_common.validate_cmp('int32', output1, ans1)
