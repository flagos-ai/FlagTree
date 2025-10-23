# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import triton
import triton.language as tl

import torch
import torch_npu
import pytest
import test_common
from triton.runtime.libentry import libentry


@pytest.mark.parametrize(
    "para_type,data_type,shape",
    [
        ["float32", torch.float32, (3, 11, 17)],
        ["float16", torch.float16, (3, 11, 17)],
        ["int8", torch.int8, (3, 11, 17)],
    ],
)
def test_flip(para_type, data_type, shape):

    def torch_func(x):
        return torch.flip(x, dims=(2,))

    @libentry()
    @triton.jit
    def triton_kernel(
        output_ptr0, in_ptr0, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr
    ):
        xidx = tl.arange(0, XB)
        yidx = tl.arange(0, YB)
        zidx = tl.arange(0, ZB)
        idx = (
            xidx[:, None, None] * YB * ZB
            + yidx[None, :, None] * ZB
            + zidx[None, None, :]
        )
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = tl.flip(tmp0, 2)
        tl.store(output_ptr0 + idx, tmp1)

    def triton_func(x):
        XB, YB, ZB = shape
        y = torch.empty_like(x)
        triton_kernel[1, 1, 1](y, x, XB, YB, ZB)
        return y

    x = torch.randint(low=-128, high=128, size=shape, dtype=data_type).npu()
    torch_ref = torch_func(x)
    triton_cal = triton_func(x)
    test_common.validate_cmp(para_type, torch_ref, triton_cal)
