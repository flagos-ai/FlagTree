# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import logging
import pytest
import torch
import torch_npu
import triton
import triton.language as tl

import test_common
from test_common import TestUtils, check_ub_mem_overflow


@triton.jit
def fn_npu_0d(output_ptr, x_ptr, YB: tl.constexpr):
    yidx = tl.arange(0, YB)

    idx = yidx

    X = tl.load(x_ptr + idx)

    ret = tl.zeros_like(X)

    oidx = yidx

    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_1d(output_ptr, x_ptr, YB: tl.constexpr):
    yidx = tl.arange(0, YB)

    idx = yidx

    X = tl.load(x_ptr + idx)

    ret = tl.zeros_like(X)

    oidx = yidx

    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_2d(output_ptr, x_ptr, YB: tl.constexpr, ZB: tl.constexpr):
    pid = tl.program_id(0)
    yidx = tl.arange(0, YB)[:, None] + pid * YB
    zidx = tl.arange(0, ZB)[None, :]

    idx = yidx * ZB + zidx

    X = tl.load(x_ptr + idx)

    ret = tl.zeros_like(X)

    oidx = yidx * ZB + zidx

    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_3d(output_ptr, x_ptr, YB: tl.constexpr, ZB: tl.constexpr, KB: tl.constexpr):
    yidx = tl.arange(0, YB)[:, None, None] * ZB * KB
    zidx = tl.arange(0, ZB)[None, :, None] * KB
    kidx = tl.arange(0, KB)[None, None, :]

    idx = yidx + zidx + kidx

    X = tl.load(x_ptr + idx)

    ret = tl.zeros_like(X)

    oidx = yidx + zidx + kidx

    tl.store(output_ptr + oidx, ret)


test_shape0d = [()]
testlist = test_shape0d + TestUtils.test_shape1_2_3d


@pytest.mark.parametrize('shape', testlist)
@pytest.mark.parametrize('dtype', TestUtils.dtype_list)
def test_npu(shape, dtype):
    logging.debug(f'dtype:{dtype} shape:{shape}')
    if check_ub_mem_overflow(dtype, shape):
        return
    x = torch.full(shape, 0, dtype=eval('torch.' + dtype)).npu()
    triton_res = torch.empty(shape, dtype=eval('torch.' + dtype)).npu()
    torch_res = x

    if len(shape) == 0:
        fn_npu_0d[1, 1, 1](triton_res, x, 1)
    elif len(shape) == 1:
        fn_npu_1d[1, 1, 1](triton_res, x, shape[0])
    elif len(shape) == 2:
        fn_npu_2d[shape[0], 1, 1](triton_res, x, 1, shape[1])
    elif len(shape) == 3:
        fn_npu_3d[1, 1, 1](triton_res, x, shape[0], shape[1], shape[2])

    test_common.validate_cmp(dtype, triton_res, torch_res)


@triton.jit
def fn_npu_multi_d(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr,
                   NB: tl.constexpr):
    offsets = tl.arange(0, XB) * (YB * ZB * MB * NB)
    if (YB * ZB * MB * NB) > 1:
        offsets = offsets[:, None] + tl.arange(0, YB)[None, :] * (ZB * MB * NB)
    if (ZB * MB * NB) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, ZB)[None, None, :] * (MB * NB)
    if (MB * NB) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, MB)[None, None, None, :] * NB
    if NB > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, NB)[None, None, None, None, :]

    X = tl.load(x_ptr + offsets)
    ret = tl.zeros_like(X)

    tl.store(output_ptr + offsets, ret)


@pytest.mark.shape_4d_5d
@pytest.mark.parametrize('param_list',
                         [
                             ('float32', (4, 2, 16, 16)),
                             ('float32', (2, 4, 2, 16, 16)),

                             ('float32', (4, 2, 16, 16)),
                             ('float32', (2, 4, 2, 16, 16)),

                             ('float32', (4, 2, 16, 16)),
                             ('float32', (2, 4, 2, 16, 16)),
                         ]
                         )
def test_case_4d_5d(param_list):
    dtype, shape = param_list
    if check_ub_mem_overflow(dtype, shape):
        return
    x0 = test_common.generate_tensor(shape, dtype)
    y_ref = torch.zeros_like(x0, dtype=eval('torch.' + dtype)).npu()
    print(f"y_ref = {torch.flatten(y_ref)[0:4]}")
    y_cal = torch.ones(shape, dtype=eval('torch.' + dtype)).npu()

    triton_shape = [*shape]
    while len(triton_shape) < 5:
        triton_shape.append(1)
    fn_npu_multi_d[(1,)](y_cal, x0, *triton_shape)
    print(f"y_cal = {torch.flatten(y_cal)[0:4]}")
    test_common.validate_cmp(dtype, y_cal, y_ref)


if __name__ == "__main__":
    for dtype in TestUtils.dtype_list:
        for shape in [(37,), (37, 3), (1, 22, 39)]:
            test_npu(shape, dtype)
