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
def triton_trans_4d(
        output_ptr, x_ptr, PERM: tl.constexpr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr, BLOCK_3: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr, SHAPE_3: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr, STRIDE_3: tl.constexpr,
):
    pid = tl.program_id(0)
    tmp0 = tl.arange(0, BLOCK_0)[:, None, None, None]
    tmp0_1 = tl.arange(0, BLOCK_0)[None, :, None, None]
    tmp1 = tl.arange(0, BLOCK_1)[None, :, None, None]
    tmp1_0 = tl.arange(0, BLOCK_1)[:, None, None, None]
    tmp1_2 = tl.arange(0, BLOCK_1)[None, None, :, None]
    tmp2 = tl.arange(0, BLOCK_2)[None, None, :, None]
    tmp2_1 = tl.arange(0, BLOCK_2)[None, :, None, None]
    tmp2_3 = tl.arange(0, BLOCK_2)[None, None, None, :]
    tmp3 = tl.arange(0, BLOCK_3)[None, None, None, :]
    tmp3_2 = tl.arange(0, BLOCK_3)[None, None, :, None]
    offsets = pid + tmp0 * STRIDE_0 + tmp1 * STRIDE_1 + tmp2 * STRIDE_2 + tmp3 * STRIDE_3
    masks = (tmp0 < SHAPE_0) & (tmp1 < SHAPE_1) & (tmp2 < SHAPE_2) & (tmp3 < SHAPE_3)
    x_val = tl.load(x_ptr + offsets, masks)

    if PERM == 0:  # 1, 0, 2, 3
        ret = tl.trans(x_val, (1, 0, 2, 3))
        shape0 = SHAPE_1
        shape1 = SHAPE_0
        shape2 = SHAPE_2
        shape3 = SHAPE_3
    elif PERM == 1:  # 0, 2, 1, 3
        ret = tl.trans(x_val, (0, 2, 1, 3))
        shape0 = SHAPE_0
        shape1 = SHAPE_2
        shape2 = SHAPE_1
        shape3 = SHAPE_3
    else:  # 0, 1, 3, 2
        ret = tl.trans(x_val, (0, 1, 3, 2))
        shape0 = SHAPE_0
        shape1 = SHAPE_1
        shape2 = SHAPE_3
        shape3 = SHAPE_2

    s3 = 1
    s2 = s3 * shape3
    s1 = s2 * shape2
    s0 = s1 * shape1

    if PERM == 0:  # 1, 0, 2, 3
        out_offsets = pid + tmp1_0 * s0 + tmp0_1 * s1 + tmp2 * s2 + tmp3 * s3
        out_masks = (tmp1_0 < shape0) & (tmp0_1 < shape1) & (tmp2 < shape2) & (tmp3 < shape3)
    elif PERM == 1:  # 0, 2, 1, 3
        out_offsets = pid + tmp0 * s0 + tmp2_1 * s1 + tmp1_2 * s2 + tmp3 * s3
        out_masks = (tmp0 < shape0) & (tmp1_2 < shape2) & (tmp2_1 < shape1) & (tmp3 < shape3)
    else:  # 0, 1, 3, 2
        out_offsets = pid + tmp0 * s0 + tmp1 * s1 + tmp3_2 * s2 + tmp2_3 * s3
        out_masks = (tmp0 < shape0) & (tmp1 < shape1) & (tmp3_2 < shape2) & (tmp2_3 < shape3)
    tl.store(output_ptr + out_offsets, ret, mask=out_masks)


@triton.jit
def triton_trans_5d(
        output_ptr, x_ptr, PERM: tl.constexpr,
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

    tmp0_1 = tl.arange(0, BLOCK_0)[None, :, None, None, None]
    tmp1_0 = tl.arange(0, BLOCK_1)[:, None, None, None, None]

    tmp1_2 = tl.arange(0, BLOCK_1)[None, None, :, None, None]
    tmp2_1 = tl.arange(0, BLOCK_2)[None, :, None, None, None]

    tmp2_3 = tl.arange(0, BLOCK_2)[None, None, None, :, None]
    tmp3_2 = tl.arange(0, BLOCK_3)[None, None, :, None, None]

    tmp3_4 = tl.arange(0, BLOCK_3)[None, None, None, None, :]
    tmp4_3 = tl.arange(0, BLOCK_4)[None, None, None, :, None]

    offsets = pid + tmp0 * STRIDE_0 + tmp1 * STRIDE_1 + tmp2 * STRIDE_2 + tmp3 * STRIDE_3 + tmp4 * STRIDE_4
    masks = (tmp0 < SHAPE_0) & (tmp1 < SHAPE_1) & (tmp2 < SHAPE_2) & (tmp3 < SHAPE_3) & (tmp4 < SHAPE_4)
    x_val = tl.load(x_ptr + offsets, masks)

    if PERM == 0:  # 1, 0, 2, 3, 4
        ret = tl.trans(x_val, 1, 0, 2, 3, 4)
        shape0 = SHAPE_1
        shape1 = SHAPE_0
        shape2 = SHAPE_2
        shape3 = SHAPE_3
        shape4 = SHAPE_4
    elif PERM == 1:  # 0, 2, 1, 3, 4
        ret = tl.trans(x_val, 0, 2, 1, 3, 4)
        shape0 = SHAPE_0
        shape1 = SHAPE_2
        shape2 = SHAPE_1
        shape3 = SHAPE_3
        shape4 = SHAPE_4
    elif PERM == 2:  # 0, 1, 3, 2, 4
        ret = tl.trans(x_val, 0, 1, 3, 2, 4)
        shape0 = SHAPE_0
        shape1 = SHAPE_1
        shape2 = SHAPE_3
        shape3 = SHAPE_2
        shape4 = SHAPE_4
    else:  # 0, 1, 2, 4, 3
        ret = tl.trans(x_val, 0, 1, 2, 4, 3)
        shape0 = SHAPE_0
        shape1 = SHAPE_1
        shape2 = SHAPE_2
        shape3 = SHAPE_4
        shape4 = SHAPE_3

    s4 = 1
    s3 = s4 * shape4
    s2 = s3 * shape3
    s1 = s2 * shape2
    s0 = s1 * shape1

    if PERM == 0:  # 1, 0, 2, 3, 4
        out_offsets = pid + tmp1_0 * s0 + tmp0_1 * s1 + tmp2 * s2 + tmp3 * s3 + tmp4 * s4
        out_masks = (tmp1_0 < shape0) & (tmp0_1 < shape1) & (tmp2 < shape2) & (tmp3 < shape3) & (tmp4 < shape4)
    elif PERM == 1:  # 0, 2, 1, 3, 4
        out_offsets = pid + tmp0 * s0 + tmp2_1 * s1 + tmp1_2 * s2 + tmp3 * s3 + tmp4 * s4
        out_masks = (tmp0 < shape0) & (tmp1_2 < shape2) & (tmp2_1 < shape1) & (tmp3 < shape3) & (tmp4 < shape4)
    elif PERM == 2:  # 0, 1, 3, 2, 4
        out_offsets = pid + tmp0 * s0 + tmp1 * s1 + tmp3_2 * s2 + tmp2_3 * s3 + tmp4 * s4
        out_masks = (tmp0 < shape0) & (tmp1 < shape1) & (tmp3_2 < shape2) & (tmp2_3 < shape3) & (tmp4 < shape4)
    else:  # 0, 1, 2, 4, 3
        out_offsets = pid + tmp0 * s0 + tmp1 * s1 + tmp2 * s2 + tmp4_3 * s3 + tmp3_4 * s4
        out_masks = (tmp0 < shape0) & (tmp1 < shape1) & (tmp2 < shape2) & (tmp4_3 < shape3) & (tmp3_4 < shape4)
    tl.store(output_ptr + out_offsets, ret, mask=out_masks)


@pytest.mark.parametrize('shape', TestUtils.test_shape4d + TestUtils.test_shape5d)
@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16'])
@pytest.mark.parametrize('perm', [0, 1, 2, 3])  # 4d: support 3 mode; 5d: support 4 mode
def test_trans_4d_5d(shape, dtype, perm):
    logging.log(logging.DEBUG, f"shape = {shape}")
    x = torch.randint(low=0, high=2, size=shape, dtype=eval('torch.' + dtype)).npu()
    grid = (1,)
    if len(shape) == 4:
        blocks = list(x.size())
        strides = list(x.stride())
        if perm == 0:  # 1, 0, 2, 3; exchange axis 0, 1
            output = torch.empty((shape[1], shape[0], shape[2], shape[3]), dtype=eval('torch.' + dtype)).npu()
            ans_4d = torch.permute(x, (1, 0, 2, 3))
            triton_trans_4d[grid](output, x, perm, *blocks, *blocks, *strides)
            test_common.validate_cmp(dtype, ans_4d, output)
        elif perm == 1:  # 0, 2, 1, 3; exchange axis 1, 2
            output = torch.empty((shape[0], shape[2], shape[1], shape[3]), dtype=eval('torch.' + dtype)).npu()
            ans_4d = torch.permute(x, (0, 2, 1, 3))
            triton_trans_4d[grid](output, x, perm, *blocks, *blocks, *strides)
            test_common.validate_cmp(dtype, ans_4d, output)
        elif perm == 2:  # 0, 1, 3, 2; exchange axis 2, 3
            output = torch.empty((shape[0], shape[1], shape[3], shape[2]), dtype=eval('torch.' + dtype)).npu()
            ans_4d = torch.permute(x, (0, 1, 3, 2))
            triton_trans_4d[grid](output, x, perm, *blocks, *blocks, *strides)
            test_common.validate_cmp(dtype, ans_4d, output)
        else:
            pass
    else:
        blocks = list(x.size())
        strides = list(x.stride())
        
        if perm == 0:  # 1, 0, 2, 3, 4; exchange axis 0, 1
            output = torch.empty((shape[1], shape[0], shape[2], shape[3], shape[4]), dtype=eval('torch.' + dtype)).npu()
            ans_5d = torch.permute(x, (1, 0, 2, 3, 4))
        elif perm == 1:  # 0, 2, 1, 3, 4; exchange axis 1, 2
            output = torch.empty((shape[0], shape[2], shape[1], shape[3], shape[4]), dtype=eval('torch.' + dtype)).npu()
            ans_5d = torch.permute(x, (0, 2, 1, 3, 4))
        elif perm == 2:  # 0, 1, 3, 2, 4; exchange axis 2, 3
            output = torch.empty((shape[0], shape[1], shape[3], shape[2], shape[4]), dtype=eval('torch.' + dtype)).npu()
            ans_5d = torch.permute(x, (0, 1, 3, 2, 4))
        else:  # 0, 1, 2, 4, 3; exchange axis 3, 4
            output = torch.empty((shape[0], shape[1], shape[2], shape[4], shape[3]), dtype=eval('torch.' + dtype)).npu()
            ans_5d = torch.permute(x, (0, 1, 2, 4, 3))
        triton_trans_5d[grid](output, x, perm, *blocks, *blocks, *strides)
        test_common.validate_cmp(dtype, ans_5d, output)
