# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import pytest

import triton
import triton.language as tl

import torch
import torch_npu
import test_common


def torch_divRn(x0, x1):
    return x0 / x1

@triton.jit
def triton_divRn(in_ptr0, in_ptr1, out_ptr0, XBLOCK : tl.constexpr, XBLOCK_SUB : tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop1 in range(loops1):
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = tl.load(in_ptr1 + (x0), None)
        tmp2 = tl.div_rn(tmp0, tmp1)
        tl.store(out_ptr0 + (x0), tmp2, None)

@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (2, 4096, 8), 32, 2048, 64],
                         ])

def test_divRn(param_list):
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    x1 = test_common.generate_tensor(shape, dtype)
    x2 = x1.masked_fill(x1 == 0, 1)
    x2 = x2.npu()
    y_ref = torch_divRn(x0, x2)
    y_cal = torch.zeros(shape, dtype = eval('torch.' + dtype)).npu()
    triton_divRn[ncore, 1, 1](x0, x2, y_cal, xblock, xblock_sub)
    test_common.validate_cmp(dtype, y_cal, y_ref)