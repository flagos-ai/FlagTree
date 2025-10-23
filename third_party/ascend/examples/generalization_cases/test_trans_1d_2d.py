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

    ret = tl.trans(X, 0)

    tl.store(output_ptr + idx, ret)

@pytest.mark.parametrize('shape', TestUtils.test_shape1d)
@pytest.mark.parametrize('dtype', TestUtils.dtype_list)
def test_trans_1d(shape, dtype):
    logging.debug(f'dtype:{dtype} shape:{shape}')

    data_type = eval('torch.' + dtype)
    x = torch.randint(low=0, high=2, size=shape, dtype=data_type).npu()

    triton_res = torch.randint(1, shape, dtype=data_type).npu()
    torch_res = torch.permute(x, (0,))
    fn_npu_1d[1, 1, 1](triton_res, x, shape[0])
    test_common.validate_cmp(dtype, triton_res, torch_res)

@triton.jit
def fn_npu_021(output_ptr, x_ptr, YB: tl.constexpr, ZB: tl.constexpr):
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)
    idx = yidx[:, None] * ZB + zidx[None, :]

    # XB,YB,1
    X = tl.load(x_ptr + idx)

    ret = tl.trans(X, 1, 0)

    oidx = zidx[:, None] * YB + yidx[None, :]

    tl.store(output_ptr + oidx, ret)

bisheng_notsupport_dtype = ['int64']
tritonascend_notsupport_dtype = ['bool']
# check_ub_mem_overflow没拦住，在kernel中最大ub占用超过ubsize
mem_overflow_scene = [
('bfloat16', (128, 256)), 
('bfloat16', (256, 128)),
('int8', (741,256)),
('int8', (256,741)),
('int16', (256,256)),
('float16', (256,256)),
('bfloat16', (256,256)),
('int32', (128, 256)),
('int32', (256, 128)),
('float32', (128,256)),
('float32', (256,128)),
]
@pytest.mark.parametrize('shape', TestUtils.test_shape2d)
@pytest.mark.parametrize('dtype', TestUtils.dtype_list)
def test_permute(shape, dtype):
    logging.debug(f'dtype:{dtype} shape:{shape}')
    if dtype in bisheng_notsupport_dtype or dtype in tritonascend_notsupport_dtype:
        return
    if (dtype, shape) in mem_overflow_scene:
        return
    if check_ub_mem_overflow(dtype, shape):
        return
    YB = shape[0]
    ZB = shape[1]
    data_type = eval('torch.' + dtype)
    x = torch.randint(low=0, high=2, size=(YB, ZB), dtype=data_type).npu()

    triton_res = torch.randint(1, (ZB, YB), dtype=data_type).npu()
    torch_res = torch.permute(x, (1, 0))
    fn_npu_021[1, 1, 1](triton_res, x, YB, ZB)
    test_common.validate_cmp(dtype, triton_res, torch_res)

if __name__ == "__main__":
    for shape in [(37, 3)]:
        for dtype in TestUtils.dtype_list:
            test_permute(shape, dtype)
