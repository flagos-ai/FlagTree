# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import triton
import triton.language as tl
import logging
import torch
import pytest
import test_common
from test_common import TestUtils
import math


@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr,
            XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
            XNUMEL: tl.constexpr, YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs

    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]

    X = tl.load(x_ptr + idx)

    ret = tl.view(X, (ZB * YB * XB,))

    oidx = tl.arange(0, XB * YB * ZB) + xoffs * YNUMEL * ZNUMEL + yoffs * ZNUMEL + zoffs

    tl.store(output_ptr + oidx, ret)


@triton.jit
def triton_view_4d_5d(
        output_ptr, x_ptr,
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
    ret = tl.view(x_val, (SHAPE_0 * SHAPE_1 * SHAPE_2 * SHAPE_3 * SHAPE_4,))

    pid0 = tl.program_id(0)

    flat_idx = tl.arange(0, BLOCK_0 * BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4)
    out_offsets = pid0 * BLOCK_0 * BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4 + flat_idx
    out_masks = out_offsets < SHAPE_0 * SHAPE_1 * SHAPE_2 * SHAPE_3 * SHAPE_4
    tl.store(output_ptr + out_offsets, ret, mask=out_masks)


@pytest.mark.parametrize('shape', TestUtils.full_shape)
@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
def test_view(shape, dtype):
    logging.log(logging.DEBUG, f"shape = {shape}")
    x = torch.full(shape, 100, dtype=eval('torch.' + dtype)).npu()
    y = torch.full(shape, 30, dtype=eval('torch.' + dtype)).npu()
    new_shape = (x.numel(),)

    output = torch.randint(1, new_shape, dtype=eval('torch.' + dtype)).npu()
    output1 = output
    logging.log(logging.DEBUG, f"output.dtype={output.dtype}")

    ans = x.view(new_shape)

    if len(shape) == 1:
        XB = 1;
        xnumel = 1
        YB = 1;
        ynumel = 1
        ZB = shape[0];
        znumel = shape[0]
    elif len(shape) == 2:
        XB = 1;
        xnumel = 1
        YB = shape[0];
        ynumel = shape[0]
        ZB = shape[1];
        znumel = shape[1]
    else:
        XB = shape[0];
        xnumel = shape[0]
        YB = shape[1];
        ynumel = shape[1]
        ZB = shape[2];
        znumel = shape[2]

    grid = (1, 1, 1)
    if x.numel() * x.element_size() >= 8192:
        if xnumel > 1:
            grid = (XB, 1, 1)
            XB = 1
        elif ynumel > 1:
            grid = (1, YB, 1)
            YB = 1
        else:
            grid = (1, 1, ZB)
            ZB = 1

    fn_npu_[grid](output, x, y, XB, YB, ZB, xnumel, ynumel, znumel)

    test_common.validate_cmp(dtype, ans, output)


@pytest.mark.parametrize('shape', TestUtils.test_shape4d + TestUtils.test_shape5d)
@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
def test_view_4d_5d(shape, dtype):
    logging.log(logging.DEBUG, f"shape = {shape}")
    x = torch.full(shape, 100, dtype=eval('torch.' + dtype)).npu()

    output = torch.randint(1, (x.numel(),), dtype=eval('torch.' + dtype)).npu()
    logging.log(logging.DEBUG, f"output.dtype={output.dtype}")

    ans = x.view(x.numel(), )

    blocks = list(x.size())
    strides = list(x.stride())
    while len(blocks) < 5:
        blocks.append(1)
        strides.append(1)

    grid = (1,)
    triton_view_4d_5d[grid](output, x, *blocks, *blocks, *strides)

    test_common.validate_cmp(dtype, ans, output)
