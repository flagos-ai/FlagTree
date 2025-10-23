# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import pytest

import triton
import triton.language as tl
import time
import test_common
import os
import shutil

import torch
import torch_npu


def standard_floor(x0):
    res = torch.floor(x0)
    return res


@triton.jit
def triton_floor(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    mask = idx_block < N
    x = tl.load(in_ptr0 + idx_block, mask=mask)
    res = tl.floor(x)
    tl.store(out_ptr0 + idx_block, res, mask=mask)


types = [
    (torch.float32, 'float32'),
]

# if shape axis = 32/256 , then actual shape = axis/element_size()
shapes = [
    (3, 32),
    (-32, 32),
    (37, 64),
    (-256, 256),
    (781, 1024),
]

map_for_64_t = {37: 31}

ops = [
    ('floor', triton_floor, standard_floor),
]


@pytest.mark.parametrize('opName, tritonOp, standOp', ops)
@pytest.mark.parametrize('dtype, sigtype', types)
@pytest.mark.parametrize('N, NUMEL', shapes)
def test_elementwise_common(opName, tritonOp, standOp, dtype, sigtype, N, NUMEL):
    torch.manual_seed(0)
    torch_npu.npu.utils.set_device(0)
    N = (-N) // torch.tensor(0, dtype=dtype).element_size() if N < 0 else N

    if sigtype == 'int64':
        N = map_for_64_t[N] if N in map_for_64_t else N

    x0 = test_common.generate_tensor(shape=(N,), dtype=sigtype)

    ans = standOp(x0)
    x0 = x0.npu()

    output = torch.zeros((N,), dtype=dtype).npu()
    tritonOp[1, 1, 1](x0, output, N=N, NUMEL=NUMEL, debug=True)
    test_common.validate_cmp(sigtype, output, ans)
