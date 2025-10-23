# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import torch
import torch_npu
import triton
import triton.language as tl
from triton.runtime.libentry import libentry
import pytest
from test_common import generate_tensor, validate_cmp, _32bit_dtypes, _16bit_dtypes


def torch_func(x0):
    return torch.sum(x0)


@pytest.mark.parametrize("dtype", _32bit_dtypes)
@pytest.mark.parametrize("shape", [(1,), (3,), (8,), (37,), (64,), (781,)])
def test_sum(dtype, shape):

    @libentry()
    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, XBLOCK: tl.constexpr):
        idx = tl.arange(0, XBLOCK)
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = tl.sum(tmp0)
        tl.store(out_ptr0 + idx, tmp1)

    def triton_func(x0):
        out = x0[0]
        triton_kernel[1, 1, 1](out, x0, x0.numel())
        return out

    x0 = generate_tensor(shape=shape, dtype=dtype).npu()
    torch_ref = torch_func(x0)
    triton_cal = triton_func(x0)
    validate_cmp(dtype, torch_ref, triton_cal)


@triton.jit
def _reduce_combine(a, b):
    return a + b


@pytest.mark.parametrize("dtype", _32bit_dtypes)
@pytest.mark.parametrize("shape", [(1,), (3,), (8,), (37,), (64,), (781,)])
def test_reduce_sum(dtype, shape):

    @libentry()
    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, XBLOCK: tl.constexpr):
        idx = tl.arange(0, XBLOCK)
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = tl.reduce(tmp0, 0, _reduce_combine)
        tl.store(out_ptr0 + idx, tmp1)

    def triton_func(x0):
        out = x0[0]
        triton_kernel[1, 1, 1](out, x0, x0.numel())
        return out

    x0 = generate_tensor(shape=shape, dtype=dtype).npu()
    torch_ref = torch_func(x0)
    triton_cal = triton_func(x0)
    validate_cmp(dtype, torch_ref, triton_cal)
