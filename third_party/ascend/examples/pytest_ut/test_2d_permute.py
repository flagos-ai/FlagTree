# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import pytest

import triton
import triton.language as tl

import torch
import torch_npu

def fn(x):
    return x.t()

@triton.jit
def triton_2d_permute(output_ptr, input_ptr, X : tl.constexpr, Y : tl.constexpr):
    xindex = tl.arange(0, X * Y)
    input_local = tl.load(input_ptr + xindex)
    output_local = input_local.reshape(X, Y).trans().reshape(X*Y)
    tl.store(output_ptr + xindex, output_local)


@pytest.mark.parametrize('X', [32, 64, 256])
@pytest.mark.parametrize('Y', [16, 32])
def test_cases(X, Y):

    x = torch.randn((X, Y)).npu()
    output1 = fn(x)
    output2 = torch.randn(output1.shape, dtype=output1.dtype).npu()

    triton_2d_permute[1, 1, 1](output2, x, X, Y, debug=True)
    print(output1)
    print(output2)

    torch.testing.assert_close(output1, output2, rtol=1e-3, atol=1e-3)






