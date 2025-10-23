# -*- coding: utf-8 -*-
# # Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import triton
import triton.language as tl
import torch
import pytest
import test_common
from test_common import TestUtils, check_ub_mem_overflow
import math
import logging

@triton.jit
def fn_npu_1d(output_ptr, x_ptr, xnumel:tl.constexpr):
    idx = tl.arange(0, xnumel)

    X = tl.load(x_ptr + idx)

    ret = tl.permute(X, (0,))

    tl.store(output_ptr + idx, ret)

@pytest.mark.parametrize('shape', TestUtils.test_shape1d)
@pytest.mark.parametrize('dtype', TestUtils.dtype_list)
def test_permute_1d(shape, dtype):
    logging.debug(f'dtype:{dtype} shape:{shape}')

    data_type = eval('torch.' + dtype)
    x = torch.randint(low=0, high=2, size=shape, dtype=data_type).npu()

    triton_res = torch.randint(1, shape, dtype=data_type).npu()
    torch_res = torch.permute(x, (0,))
    fn_npu_1d[1, 1, 1](triton_res, x, shape[0])
    test_common.validate_cmp(dtype, triton_res, torch_res)


@triton.jit
def fn_npu_021(output_ptr, x_ptr, YB: tl.constexpr, ZB: tl.constexpr, ynumel: tl.constexpr, znumel:tl.constexpr):
    pid = tl.program_id(0)
    yidx = tl.arange(0, YB) + pid * YB
    zidx = tl.arange(0, ZB)
    idx = yidx[:, None] * znumel + zidx[None, :]

    # XB,YB,1
    X = tl.load(x_ptr + idx)

    ret = tl.permute(X, (1, 0))

    oidx = zidx[:, None] * ynumel + yidx[None, :]

    tl.store(output_ptr + oidx, ret)

@pytest.mark.parametrize('shape', TestUtils.test_shape2d)
@pytest.mark.parametrize('dtype', TestUtils.dtype_list)
def test_permute(shape, dtype):
    logging.debug(f'dtype:{dtype} shape:{shape}')
    
    ynumel=shape[0]; YB = 1
    znumel=shape[1]; ZB = shape[1]

    data_type = eval('torch.' + dtype)
    x = torch.randint(low=0, high=2, size=(shape[0], shape[1]), dtype=data_type).npu()

    triton_res = torch.randint(1, (shape[1], shape[0]), dtype=data_type).npu()
    torch_res = torch.permute(x, (1, 0))
    fn_npu_021[shape[0], 1, 1](triton_res, x, YB, ZB, ynumel, znumel)
    test_common.validate_cmp(dtype, triton_res, torch_res)

if __name__ == "__main__":
    for shape in [(37, 3)]:
        for dtype in TestUtils.dtype_list:
            test_permute(shape, dtype)
