# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Only floating point clamp is supported
import pytest

import triton
import triton.language as tl
import torch
import test_common
from test_common import TestUtils
import logging


@triton.jit
def triton_mul(output_ptr, x_ptr, y_ptr, z_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
               XNUMEL: tl.constexpr, YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs

    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]

    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)

    ret = X * Y

    tl.store(output_ptr + idx, ret)


@triton.jit
def triton_mul_4d_5d(
        output_ptr, x_ptr, y_ptr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr, BLOCK_3: tl.constexpr,
        BLOCK_4: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr, SHAPE_3: tl.constexpr,
        SHAPE_4: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr, STRIDE_3: tl.constexpr,
        STRIDE_4: tl.constexpr
):
    offsets = tl.program_id(0)

    offsets = offsets + tl.arange(0, BLOCK_0) * STRIDE_0
    masks = tl.arange(0, BLOCK_0) < SHAPE_0
    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        masks = masks[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        masks = masks[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        masks = masks[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if BLOCK_4 > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        masks = masks[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)

    x_val = tl.load(x_ptr + offsets, masks)
    y_val = tl.load(y_ptr + offsets, masks)
    ret = x_val * y_val
    tl.store(output_ptr + offsets, ret, mask=masks)


@pytest.mark.parametrize('shape', TestUtils.full_shape) # some shape with int8 over ub
@pytest.mark.parametrize('dtype', ['bool', 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16'])
def test_mul(shape, dtype):
    logging.log(logging.DEBUG, f"shape = {shape}")
    x = test_common.generate_tensor(shape, dtype).npu()
    y = test_common.generate_tensor(shape, dtype).npu()
    z = test_common.generate_tensor(shape, dtype).npu()

    ans = x * y
    output = torch.zeros_like(ans)

    if len(shape) == 1:
        triton_mul[1, 1, shape[0]](output, x, y, z, 1, 1, 1, 1, 1, shape[0])
    elif len(shape) == 2:
        if shape[0] > shape[1]:
            triton_mul[1, shape[0], 1](output, x, y, z, 1, 1, shape[1], 1, shape[0], shape[1])
        else:
            triton_mul[1, 1, shape[1]](output, x, y, z, 1, shape[0], 1, 1, shape[0], shape[1])
    elif len(shape) == 3:
        if max(shape[0], shape[1], shape[2]) == shape[0]:
            triton_mul[shape[0], 1, 1](output, x, y, z, 1, shape[1], shape[2], shape[0], shape[1], shape[2])
        elif max(shape[0], shape[1], shape[2]) == shape[1]:
            triton_mul[1, shape[1], 1](output, x, y, z, shape[0], 1, shape[2], shape[0], shape[1], shape[2])
        else:
            triton_mul[1, 1, shape[2]](output, x, y, z, shape[0], shape[1], 1, shape[0], shape[1], shape[2])
    else:
        triton_mul[1, 1, 1](output, x, y, z, 1, 1, 1, 1, 1, 1)

    test_common.validate_cmp(dtype, ans, output)


@pytest.mark.parametrize('shape', TestUtils.test_shape4d + TestUtils.test_shape5d)
@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16'])
def test_mul_4d_5d(shape, dtype):
    logging.log(logging.DEBUG, f"shape = {shape}")
    x = test_common.generate_tensor(shape, dtype).npu()
    y = test_common.generate_tensor(shape, dtype).npu()

    output = torch.randint(1, shape, dtype=eval('torch.' + dtype)).npu()

    logging.log(logging.DEBUG, f"output.dtype={output.dtype}")

    ans = x * y

    blocks = list(x.size())
    strides = list(x.stride())
    while len(blocks) < 5:
        blocks.append(1)
        strides.append(1)

    grid = (1,)
    triton_mul_4d_5d[grid](output, x, y, *blocks, *blocks, *strides)

    test_common.validate_cmp(dtype, ans, output)