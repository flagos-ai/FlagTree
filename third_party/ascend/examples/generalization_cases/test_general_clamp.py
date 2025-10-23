# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Only floating point clamp is supported
import pytest

import triton
import triton.language as tl
import time
import torch
import torch_npu
import test_common
from test_common import TestUtils
import logging


def torch_clamp(x0, min_, max_):
    res = torch.clamp(x0, min_, max_)
    return res


@triton.jit
def tt_clamp_1d(in_ptr, out_ptr, min_ptr, max_ptr,
                xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    idx = tl.arange(0, XB)

    x = tl.load(in_ptr + idx)
    min_ = tl.load(min_ptr + idx)
    max_ = tl.load(max_ptr + idx)
    ret = tl.clamp(x, min_, max_)

    tl.store(out_ptr + idx, ret)


@triton.jit
def tt_clamp_2d(in_ptr, out_ptr, min_ptr, max_ptr,
                xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    idx = xidx[:, None] * ynumel + yidx[None, :]

    x = tl.load(in_ptr + idx)
    min_ = tl.load(min_ptr + idx)
    max_ = tl.load(max_ptr + idx)
    ret = tl.clamp(x, min_, max_)

    tl.store(out_ptr + idx, ret)


@triton.jit
def tt_clamp_3d(in_ptr, out_ptr, min_ptr, max_ptr,
                xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs

    idx = xidx[:, None, None] * ynumel * znumel + yidx[None, :, None] * znumel + zidx[None, None, :]

    x = tl.load(in_ptr + idx)
    min_ = tl.load(min_ptr + idx)
    max_ = tl.load(max_ptr + idx)
    ret = tl.clamp(x, min_, max_)

    tl.store(out_ptr + idx, ret)


@triton.jit
def triton_clamp_4d_5d(
        x_ptr, output_ptr, min_ptr, max_ptr,
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
    min_ = tl.load(min_ptr + offsets)
    max_ = tl.load(max_ptr + offsets)
    ret = tl.clamp(x_val, min_, max_)
    tl.store(output_ptr + offsets, ret, mask=masks)


@pytest.mark.parametrize('shape', TestUtils.full_shape)
@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
def test_clamp(shape, dtype):
    logging.log(logging.DEBUG, f"shape = {shape}")
    torch.manual_seed(0)
    x = test_common.generate_tensor(shape, dtype).npu()
    a = test_common.generate_tensor(shape, dtype)
    b = test_common.generate_tensor(shape, dtype)
    min_ = torch.min(a, b).npu()
    max_ = torch.max(a, b).npu()

    grid = (1, 1, 1)

    y_cal = torch.empty(shape, dtype=eval('torch.' + dtype), device="npu")

    y_ref = torch_clamp(x, min_, max_)
    if len(shape) == 1:
        tt_clamp_1d[grid](x, y_cal, min_, max_, x.numel(), 1, 1, x.numel(), 1, 1)
    elif len(shape) == 2:
        xnumel, ynumel, znumel = shape + (1,)
        XB, YB, ZB = xnumel, ynumel, znumel
        if x.numel() * x.element_size() > 8192:
            grid = (1, ynumel, 1)
            YB = 1
        tt_clamp_2d[grid](x, y_cal, min_, max_, xnumel, ynumel, znumel, XB, YB, ZB)

    elif len(shape) == 3:
        xnumel, ynumel, znumel = shape
        XB, YB, ZB = xnumel, ynumel, znumel
        tt_clamp_3d[grid](x, y_cal, min_, max_, xnumel, ynumel, znumel, XB, YB, ZB)

    test_common.validate_cmp(dtype, y_cal, y_ref)


@pytest.mark.parametrize('shape', TestUtils.test_shape4d + TestUtils.test_shape5d)
@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
def test_clamp_4d_5d(shape, dtype):
    logging.log(logging.DEBUG, f"shape = {shape}")
    torch.manual_seed(0)
    x = test_common.generate_tensor(shape, dtype).npu()
    a = test_common.generate_tensor(shape, dtype)
    b = test_common.generate_tensor(shape, dtype)
    min_ = torch.min(a, b).npu()
    max_ = torch.max(a, b).npu()

    output = torch.empty(shape, dtype=eval('torch.' + dtype), device="npu")

    logging.log(logging.DEBUG, f"output.dtype={output.dtype}")

    ans = torch_clamp(x, min_, max_)

    blocks = list(x.size())
    strides = list(x.stride())
    while len(blocks) < 5:
        blocks.append(1)
        strides.append(1)

    grid = (1,)
    triton_clamp_4d_5d[grid](x, output, min_, max_, *blocks, *blocks, *strides)

    test_common.validate_cmp(dtype, ans, output)
