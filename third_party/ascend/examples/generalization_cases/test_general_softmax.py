# 实际实现与官网定义不符，可能和triton submodule版本有关， 当前的submodule 不接受指定dim，都是按第0维做softmax
# arith.maximum 不支持类似 1x3 -> 3 和 1 -> 1 的reduce
import triton
import triton.language as tl
import torch
import logging
import pytest
import test_common
from test_common import TestUtils
import math


def torch_softmax_d0(x1):
    res = torch.softmax(x1, axis=0).to(x1.dtype)
    return res


@triton.jit
def tt_softmax_1d(in_ptr, out_ptr,
                  xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                  XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    idx = tl.arange(0, XB)
    x = tl.load(in_ptr + idx)
    ret = tl.softmax(x)
    tl.store(out_ptr + idx, ret)


@triton.jit
def tt_softmax_2d(in_ptr, out_ptr,
                  xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                  XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    idx = xidx[:, None] * ynumel + yidx[None, :]

    a = tl.load(in_ptr + idx)
    ret = tl.softmax(a)

    tl.store(out_ptr + idx, ret)


@triton.jit
def tt_softmax_3d(in_ptr, out_ptr,
                  xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                  XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs

    idx = xidx[:, None, None] * ynumel * znumel + yidx[None, :, None] * znumel + zidx[None, None, :]

    a = tl.load(in_ptr + idx)
    ret = tl.softmax(a)

    tl.store(out_ptr + idx, ret)


@triton.jit
def triton_softmax_4d_5d(
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
    ret = tl.softmax(x_val)
    tl.store(output_ptr + offsets, ret, mask=masks)


@pytest.mark.parametrize('shape', TestUtils.full_shape)
@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
def test_softmax(dtype, shape):
    logging.log(logging.DEBUG, f"shape = {shape}", flush=True)
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=eval('torch.' + dtype), device="npu") * 10
    grid = (1, 1, 1)

    y_cal = torch.rand(shape, dtype=eval('torch.' + dtype), device="npu")

    y_ref = torch_softmax_d0(x)
    if len(shape) == 1:
        tt_softmax_1d[grid](x, y_cal, x.numel(), 1, 1, x.numel(), 1, 1)
    elif len(shape) == 2:
        xnumel, ynumel, znumel = shape + (1,)
        XB, YB, ZB = xnumel, ynumel, znumel
        if x.numel() * x.element_size() > 8192:
            grid = (1, ynumel, 1)
            YB = 1
        tt_softmax_2d[grid](x, y_cal, xnumel, ynumel, znumel, XB, YB, ZB)

    elif len(shape) == 3:
        mx = max(shape[1], shape[2])
        if mx == shape[1]:
            tt_softmax_3d[1, shape[1], 1](x, y_cal, shape[0], shape[1], shape[2], shape[0], 1, shape[2])
        else:
            tt_softmax_3d[1, 1, shape[2]](x, y_cal, shape[0], shape[1], shape[2], shape[0], shape[1], 1)

    test_common.validate_cmp(dtype, y_cal, y_ref)


invalid_types = [
    'int8',
    'int16',
    'int32',
    'uint32',
    'int64',
    'bool',
]


@pytest.mark.parametrize("dtype", invalid_types)
@test_common.raises_with_match(triton.compiler.errors.CompilationError, "Expected dtype")
def test_softmax_invalid_dtype_case(dtype):
    x0 = test_common.generate_tensor((1,), dtype).npu()

    y_cal = torch.zeros((1,), dtype=eval('torch.' + dtype)).npu()
    tt_softmax_1d[1, 1, 1](x0, y_cal, 0, 0, 0, 1, 0, 0)


@pytest.mark.parametrize('shape', TestUtils.test_shape4d + TestUtils.test_shape5d)
@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
def test_softmax_4d_5d(shape, dtype):
    logging.log(logging.DEBUG, f"shape = {shape}")
    x = test_common.generate_tensor(shape, dtype).npu()

    output = torch.randint(1, shape, dtype=eval('torch.' + dtype)).npu()
    logging.log(logging.DEBUG, f"output.dtype={output.dtype}")

    ans = torch_softmax_d0(x)

    blocks = list(x.size())
    strides = list(x.stride())
    while len(blocks) < 5:
        blocks.append(1)
        strides.append(1)

    grid = (1,)
    triton_softmax_4d_5d[grid](output, x, *blocks, *blocks, *strides)

    test_common.validate_cmp(dtype, ans, output)
