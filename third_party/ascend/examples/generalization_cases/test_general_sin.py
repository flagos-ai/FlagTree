import triton
import triton.language as tl
import torch
import numpy as np
import pytest
import test_common
from test_common import TestUtils
import math


@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr, z_ptr,
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

    ret = tl.sin(X)

    tl.store(output_ptr + idx, ret)


@triton.jit
def triton_sin_4d_5d(
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
    ret = tl.sin(x_val)
    tl.store(output_ptr + offsets, ret, mask=masks)


import logging


@pytest.mark.parametrize('shape', TestUtils.full_shape)
@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
def test_sin(dtype, shape, ):
    x = test_common.generate_tensor(shape, dtype).npu()
    y = test_common.generate_tensor(shape, dtype).npu()
    z = test_common.generate_tensor(shape, dtype).npu()
    new_shape = shape

    output = torch.randint(1, new_shape, dtype=eval('torch.' + dtype)).npu()
    output1 = output
    logging.log(logging.DEBUG, f"output.dtype={output.dtype}")

    ans = torch.sin(x)

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

    fn_npu_[grid](output, x, y, z, XB, YB, ZB, xnumel, ynumel, znumel)

    test_common.validate_cmp(dtype, ans, output)


invalid_dtypes = [
    'int8',
    'int16',
    'int32',
    'uint32',
    'int64',
    'bool',
]


@pytest.mark.parametrize("dtype", invalid_dtypes)
@test_common.raises_with_match(triton.compiler.errors.CompilationError, "Expected dtype")
def test_sin_invalid_dtype_case(dtype):
    x = test_common.generate_tensor((1,), dtype).npu()
    y = test_common.generate_tensor((1,), dtype).npu()
    z = test_common.generate_tensor((1,), dtype).npu()

    output = torch.randint(1, (1,), dtype=eval('torch.' + dtype)).npu()
    fn_npu_[1, 1, 1](output, x, y, z, 1, 1, 1, 1, 1, 1)


@pytest.mark.parametrize('shape', TestUtils.test_shape4d + TestUtils.test_shape5d)
@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
def test_sin_4d_5d(shape, dtype):
    logging.log(logging.DEBUG, f"shape = {shape}")
    x = test_common.generate_tensor(shape, dtype).npu()
    y = test_common.generate_tensor(shape, dtype).npu()

    output = torch.randint(1, shape, dtype=eval('torch.' + dtype)).npu()

    logging.log(logging.DEBUG, f"output.dtype={output.dtype}")

    ans = torch.sin(x)


    blocks = list(x.size())
    strides = list(x.stride())
    while len(blocks) < 5:
        blocks.append(1)
        strides.append(1)

    grid = (1,)
    triton_sin_4d_5d[grid](output, x, *blocks, *blocks, *strides)

    test_common.validate_cmp(dtype, ans, output)
