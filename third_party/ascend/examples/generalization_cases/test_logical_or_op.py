# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import pytest
import triton
import triton.language as tl
import time
import torch
import torch_npu
import test_common
from test_common import TestUtils, generate_tensor
import logging


@triton.jit
def triton_logical_or_1d(in_ptr0, in_ptr1, out_ptr0, L: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    idx = lblk_idx
    x0 = tl.load(in_ptr0 + idx)
    x1 = tl.load(in_ptr1 + idx)
    ret = x0.logical_or(x1)
    odx = lblk_idx
    tl.store(out_ptr0 + odx, ret)


@triton.jit
def triton_logical_or_2d(in_ptr0, in_ptr1, out_ptr0, L: tl.constexpr, M: tl.constexpr):
    pid = tl.program_id(0)
    lblk_idx = tl.arange(0, L) + pid * L
    mblk_idx = tl.arange(0, M)
    idx = lblk_idx[:, None] * M + mblk_idx[None, :]
    x0 = tl.load(in_ptr0 + idx)
    x1 = tl.load(in_ptr1 + idx)
    ret = x0.logical_or(x1)
    odx = lblk_idx[:, None] * M + mblk_idx[None, :]
    tl.store(out_ptr0 + odx, ret)


@triton.jit
def triton_logical_or_3d(in_ptr0, in_ptr1, out_ptr0, XB, YB, ZB, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    lblk_idx = tl.arange(0, L) + tl.program_id(0) * XB
    mblk_idx = tl.arange(0, M) + tl.program_id(1) * YB
    nblk_idx = tl.arange(0, N) + tl.program_id(2) * ZB
    idx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    x0 = tl.load(in_ptr0 + idx)
    x1 = tl.load(in_ptr1 + idx)
    ret = x0.logical_or(x1)
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    tl.store(out_ptr0 + odx, ret)


@triton.jit
def triton_logical_or_4d_5d(
        x_ptr, y_ptr, output_ptr,
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
    ret = x_val.logical_or(y_val)
    tl.store(output_ptr + offsets, ret, mask=masks)


support_typelist = ['bool', ]


@pytest.mark.parametrize('shape', TestUtils.full_shape)
@pytest.mark.parametrize('sigtype', support_typelist)
def test_logical_or(shape, sigtype):
    logging.debug(f"dtype:{sigtype} shape:{shape}")
    dtype = eval('torch.' + sigtype)
    x0 = generate_tensor(shape=shape, dtype=sigtype).npu()
    x1 = generate_tensor(shape=shape, dtype=sigtype).npu()
    # ncore, xblock, xblock_sub = 2, 32768, 1024
    y_ref = torch.logical_or(x0, x1)
    output = torch.zeros(shape, dtype=dtype).npu()
    if len(shape) == 1:
        triton_logical_or_1d[1, 1, 1](x0, x1, output, shape[0])
    elif len(shape) == 2:
        shape0 = shape[0]
        shape1 = shape[1]
        if x0.numel() * x0.element_size() >= 8192:
            grid = (shape0, 1, 1)
            shape0 = 1
        else:
            grid = (1, 1, 1)
        triton_logical_or_2d[grid](x0, x1, output, shape0, shape1)
    elif len(shape) == 3:
        mx = max(shape[0], shape[1], shape[2])
        if mx == shape[0]:
            triton_logical_or_3d[shape[0], 1, 1](x0, x1, output, 1, shape[1], shape[2], shape[0], shape[1], shape[2])
        elif mx == shape[1]:
            triton_logical_or_3d[1, shape[1], 1](x0, x1, output, shape[0], 1, shape[2], shape[0], shape[1], shape[2])
        else:
            triton_logical_or_3d[1, 1, shape[2]](x0, x1, output, shape[0], shape[1], 1, shape[0], shape[1], shape[2])
    test_common.validate_cmp(sigtype, output, y_ref)


@pytest.mark.parametrize('shape', TestUtils.test_shape4d + TestUtils.test_shape5d)
@pytest.mark.parametrize('dtype', ['bool'])
def test_logical_or_4d_5d(shape, dtype):
    logging.log(logging.DEBUG, f"shape = {shape}")
    x = test_common.generate_tensor(shape, dtype).npu()
    y = test_common.generate_tensor(shape, dtype).npu()

    output = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    logging.log(logging.DEBUG, f"output.dtype={output.dtype}")

    ans = torch.logical_or(x, y)

    blocks = list(x.size())
    strides = list(x.stride())
    while len(blocks) < 5:
        blocks.append(1)
        strides.append(1)

    grid = (1,)
    triton_logical_or_4d_5d[grid](x, y, output, *blocks, *blocks, *strides)

    test_common.validate_cmp(dtype, ans, output)
