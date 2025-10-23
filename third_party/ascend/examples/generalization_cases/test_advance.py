# -*- coding: utf-8 -*-
# # Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import triton
import triton.language as tl

import torch
import torch_npu
import pytest
import test_common
from test_common import TestUtils


@triton.jit
def fn_npu_1d(output_ptr, x_ptr, y_ptr, z_ptr, output_ptr1, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    block_ptr_in = tl.make_block_ptr(
        base=x_ptr,
        shape=(XB,),
        strides=(1,),
        offsets=(5,),
        block_shape=(XB,),
        order=(0,),
    )
    bbptr = tl.advance(block_ptr_in, (-5,))
    # XB,YB,1
    X = tl.load(bbptr)

    block_ptr_out = tl.make_block_ptr(
        base=output_ptr,
        shape=(XB,),
        strides=(1,),
        offsets=(0,),
        block_shape=(XB,),
        order=(0,),
    )
    tl.store(block_ptr_out, X)


@triton.jit
def fn_npu_2d(output_ptr, x_ptr, y_ptr, z_ptr, output_ptr1, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xoffset = tl.program_id(0)
    block_ptr_in = tl.make_block_ptr(
        base=x_ptr,
        shape=(XB, YB),
        strides=(YB, 1),
        offsets=(6 + xoffset, 5),
        block_shape=(XB, YB),
        order=(1, 0),
    )
    bbptr = tl.advance(block_ptr_in, (-6, -5))
    # XB,YB,1
    X = tl.load(bbptr)

    block_ptr_out = tl.make_block_ptr(
        base=output_ptr,
        shape=(XB, YB),
        strides=(YB, 1),
        offsets=(xoffset, 0),
        block_shape=(XB, YB),
        order=(1, 0),
    )
    tl.store(block_ptr_out, X)


@triton.jit
def fn_npu_3d(output_ptr, x_ptr, y_ptr, z_ptr, output_ptr1, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    block_ptr_in = tl.make_block_ptr(
        base=x_ptr,
        shape=(XB, YB, ZB),
        strides=(YB * ZB, ZB, 1),
        offsets=(3, 1, 2),
        block_shape=(XB, YB, ZB),
        order=(2, 1, 0),
    )
    bbptr = tl.advance(block_ptr_in, (-3, -1, -2))
    # XB,YB,1
    X = tl.load(bbptr)

    block_ptr_out = tl.make_block_ptr(
        base=output_ptr,
        shape=(XB, YB, ZB),
        strides=(YB * ZB, ZB, 1),
        offsets=(0, 0, 0),
        block_shape=(XB, YB, ZB),
        order=(2, 1, 0),
    )
    tl.store(block_ptr_out, X)


@triton.jit
def triton_advance_4d(output_ptr, x_ptr,
                      BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr, BLOCK_3: tl.constexpr,
                      SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr, SHAPE_3: tl.constexpr,
                      STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr,
                      STRIDE_3: tl.constexpr, ):
    block_ptr_in = tl.make_block_ptr(
        base=x_ptr,
        shape=(SHAPE_0, SHAPE_1, SHAPE_2, SHAPE_3),
        strides=(STRIDE_0, STRIDE_1, STRIDE_2, STRIDE_3),
        offsets=(6, 5, 4, 3),
        block_shape=(BLOCK_0, BLOCK_1, BLOCK_2, BLOCK_3),
        order=(3, 2, 1, 0),
    )
    bbptr = tl.advance(block_ptr_in, (-6, -5, -4, -3))
    x = tl.load(bbptr)

    block_ptr_out = tl.make_block_ptr(
        base=output_ptr,
        shape=(SHAPE_0, SHAPE_1, SHAPE_2, SHAPE_3),
        strides=(STRIDE_0, STRIDE_1, STRIDE_2, STRIDE_3),
        offsets=(0, 0, 0, 0),
        block_shape=(BLOCK_0, BLOCK_1, BLOCK_2, BLOCK_3),
        order=(3, 2, 1, 0),
    )
    tl.store(block_ptr_out, x)


@triton.jit
def triton_advance_5d(output_ptr, x_ptr,
                      BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr, BLOCK_3: tl.constexpr,
                      BLOCK_4: tl.constexpr,
                      SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr, SHAPE_3: tl.constexpr,
                      SHAPE_4: tl.constexpr,
                      STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr,
                      STRIDE_3: tl.constexpr, STRIDE_4: tl.constexpr, ):
    block_ptr_in = tl.make_block_ptr(
        base=x_ptr,
        shape=(SHAPE_0, SHAPE_1, SHAPE_2, SHAPE_3, SHAPE_4),
        strides=(STRIDE_0, STRIDE_1, STRIDE_2, STRIDE_3, STRIDE_4),
        offsets=(6, 5, 4, 3, 2),
        block_shape=(BLOCK_0, BLOCK_1, BLOCK_2, BLOCK_3, BLOCK_4),
        order=(4, 3, 2, 1, 0),
    )
    bbptr = tl.advance(block_ptr_in, (-6, -5, -4, -3, -2))
    x = tl.load(bbptr)

    block_ptr_out = tl.make_block_ptr(
        base=output_ptr,
        shape=(SHAPE_0, SHAPE_1, SHAPE_2, SHAPE_3, SHAPE_4),
        strides=(STRIDE_0, STRIDE_1, STRIDE_2, STRIDE_3, STRIDE_4),
        offsets=(0, 0, 0, 0, 0),
        block_shape=(BLOCK_0, BLOCK_1, BLOCK_2, BLOCK_3, BLOCK_4),
        order=(4, 3, 2, 1, 0),
    )
    tl.store(block_ptr_out, x)


temporarily_not_support_dtype = ['bool']


@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
@pytest.mark.parametrize('shape', TestUtils.full_shape)
def test_npu(dtype, shape):
    if dtype in temporarily_not_support_dtype:
        return
    x = test_common.generate_tensor(shape, dtype).npu()
    y = test_common.generate_tensor(shape, dtype).npu()
    z = test_common.generate_tensor(shape, dtype).npu()

    output = torch.randint(1, shape, dtype=eval('torch.' + dtype)).npu()
    output1 = output

    a = x
    blocks = list(x.size())
    strides = list(x.stride())
    grid = (1,)
    if len(shape) == 5:
        triton_advance_5d[grid](output, x, *blocks, *blocks, *strides)
    elif len(shape) == 4:
        triton_advance_4d[grid](output, x, *blocks, *blocks, *strides)
    elif len(shape) == 3:
        fn_npu_3d[1, 1, 1](output, x, y, z, output1, XB=shape[0], YB=shape[1], ZB=shape[2])
    elif len(shape) == 2:
        if x.numel() * x.element_size() > 8192:
            fn_npu_2d[shape[0], 1, 1](output, x, y, z, output1, XB=1, YB=shape[1], ZB=1)
        else:
            fn_npu_2d[1, 1, 1](output, x, y, z, output1, XB=shape[0], YB=shape[1], ZB=1)
    else:
        fn_npu_1d[1, 1, 1](output, x, y, z, output1, XB=shape[0], YB=1, ZB=1)

    torch.testing.assert_close(output, a)
