# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import triton
import triton.language as tl
import logging
import torch
import torch_npu
import pytest
import test_common
from test_common import TestUtils


@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr,
            XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
            XNUMEL: tl.constexpr, YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB
    zoffs2 = tl.program_id(2) * ZB * 2

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs
    zidx2 = tl.arange(0, 2 * ZB) + zoffs2

    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]

    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)

    ret = tl.interleave(X, Y)

    oidx = xidx[:, None, None] * YNUMEL * ZNUMEL * 2 + yidx[None, :, None] * ZNUMEL * 2 + zidx2[None, None, :]

    tl.store(output_ptr + oidx, ret)


@triton.jit
def triton_interleave_4d(
        output_ptr, x_ptr, y_ptr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr, BLOCK_3: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr, SHAPE_3: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr, STRIDE_3: tl.constexpr,
):
    pid = tl.program_id(0)
    tmp0 = tl.arange(0, BLOCK_0)[:, None, None, None]
    tmp1 = tl.arange(0, BLOCK_1)[None, :, None, None]
    tmp2 = tl.arange(0, BLOCK_2)[None, None, :, None]
    tmp3 = tl.arange(0, BLOCK_3)[None, None, None, :]
    tmp4 = tl.arange(0, 2 * BLOCK_3)[None, None, None, :]
    offsets = pid + tmp0 * STRIDE_0 + tmp1 * STRIDE_1 + tmp2 * STRIDE_2 + tmp3 * STRIDE_3
    masks = (tmp0 < SHAPE_0) & (tmp1 < SHAPE_1) & (tmp2 < SHAPE_2) & (tmp3 < SHAPE_3)
    x_val = tl.load(x_ptr + offsets, masks)
    y_val = tl.load(y_ptr + offsets, masks)

    ret = tl.interleave(x_val, y_val)

    out_offsets = pid + tmp0 * STRIDE_0 * 2 + tmp1 * STRIDE_1 * 2 + tmp2 * STRIDE_2 * 2 + tmp4 * STRIDE_3
    out_masks = (tmp0 < SHAPE_0) & (tmp1 < SHAPE_1) & (tmp2 < SHAPE_2) & (tmp4 < 2 * SHAPE_3)
    tl.store(output_ptr + out_offsets, ret, mask=out_masks)


@triton.jit
def triton_interleave_5d(
        output_ptr, x_ptr, y_ptr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr, BLOCK_3: tl.constexpr,
        BLOCK_4: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr, SHAPE_3: tl.constexpr,
        SHAPE_4: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr, STRIDE_3: tl.constexpr,
        STRIDE_4: tl.constexpr
):
    pid = tl.program_id(0)
    tmp0 = tl.arange(0, BLOCK_0)[:, None, None, None, None]
    tmp1 = tl.arange(0, BLOCK_1)[None, :, None, None, None]
    tmp2 = tl.arange(0, BLOCK_2)[None, None, :, None, None]
    tmp3 = tl.arange(0, BLOCK_3)[None, None, None, :, None]
    tmp4 = tl.arange(0, BLOCK_4)[None, None, None, None, :]
    tmp5 = tl.arange(0, 2 * BLOCK_4)[None, None, None, None, :]
    offsets = pid + tmp0 * STRIDE_0 + tmp1 * STRIDE_1 + tmp2 * STRIDE_2 + tmp3 * STRIDE_3 + tmp4 * STRIDE_4
    masks = (tmp0 < SHAPE_0) & (tmp1 < SHAPE_1) & (tmp2 < SHAPE_2) & (tmp3 < SHAPE_3) & (tmp4 < SHAPE_4)
    x_val = tl.load(x_ptr + offsets, masks)
    y_val = tl.load(y_ptr + offsets, masks)

    ret = tl.interleave(x_val, y_val)

    out_offsets = pid + tmp0 * STRIDE_0 * 2 + tmp1 * STRIDE_1 * 2 + tmp2 * STRIDE_2 * 2 + tmp3 * STRIDE_3 * 2 + tmp5 * STRIDE_4
    out_masks = (tmp0 < SHAPE_0) & (tmp1 < SHAPE_1) & (tmp2 < SHAPE_2) & (tmp3 < SHAPE_3) & (tmp5 < 2 * SHAPE_4)
    tl.store(output_ptr + out_offsets, ret, mask=out_masks)


@pytest.mark.parametrize('shape', TestUtils.full_shape)
@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
def test_interleave(shape, dtype):
    logging.log(logging.DEBUG, f"shape = {shape}")
    x = torch.full(shape, 100, dtype=eval('torch.' + dtype)).npu()
    y = torch.full(shape, 30, dtype=eval('torch.' + dtype)).npu()
    new_shape = shape[:-1] + (2 * shape[-1],)

    output = torch.randint(1, new_shape, dtype=eval('torch.' + dtype)).npu()
    output1 = output
    logging.log(logging.DEBUG, f"output.dtype={output.dtype}")

    ans = torch.stack((x, y), dim=-1).reshape(new_shape)

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
        grid = (1, 1, ZB)
        ZB = 1

    fn_npu_[grid](output, x, y, XB, YB, ZB, xnumel, ynumel, znumel)

    test_common.validate_cmp(dtype, ans, output)


@pytest.mark.parametrize('shape', TestUtils.test_shape4d + TestUtils.test_shape5d)
@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
def test_interleave_4d_5d(shape, dtype):
    logging.log(logging.DEBUG, f"shape = {shape}")
    x = test_common.generate_tensor(shape, dtype).npu()
    y = test_common.generate_tensor(shape, dtype).npu()
    new_shape = shape[:-1] + (2 * shape[-1],)

    output = torch.randint(1, new_shape, dtype=eval('torch.' + dtype)).npu()
    logging.log(logging.DEBUG, f"output.dtype={output.dtype}")

    ans = torch.stack((x, y), dim=-1).reshape(new_shape)

    blocks = list(x.size())
    strides = list(x.stride())

    grid = (1,)
    if len(shape) == 4:
        triton_interleave_4d[grid](output, x, y, *blocks, *blocks, *strides)
    else:
        triton_interleave_5d[grid](output, x, y, *blocks, *blocks, *strides)
    test_common.validate_cmp(dtype, ans, output)
