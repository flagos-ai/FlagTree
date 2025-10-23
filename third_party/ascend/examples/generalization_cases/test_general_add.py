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
def triton_add(output_ptr, x_ptr, y_ptr, z_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
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

    ret = X + Y

    tl.store(output_ptr + idx, ret)


@triton.jit
def triton_add_broadcast(in_ptr0, in_ptr1, out_ptr0, X_SHAPE_0: tl.constexpr,
                X_SHAPE_1: tl.constexpr, X_SHAPE_2: tl.constexpr, X_SHAPE_3: tl.constexpr, X_SHAPE_4: tl.constexpr,
                Y_SHAPE_0: tl.constexpr, Y_SHAPE_1: tl.constexpr, Y_SHAPE_2: tl.constexpr, Y_SHAPE_3: tl.constexpr,
                Y_SHAPE_4: tl.constexpr):
    x_idx0 = tl.arange(0, X_SHAPE_0)
    x_idx1 = tl.arange(0, X_SHAPE_1)
    x_idx2 = tl.arange(0, X_SHAPE_2)
    x_idx3 = tl.arange(0, X_SHAPE_3)
    x_idx4 = tl.arange(0, X_SHAPE_4)

    y_idx0 = tl.arange(0, Y_SHAPE_0)
    y_idx1 = tl.arange(0, Y_SHAPE_1)
    y_idx2 = tl.arange(0, Y_SHAPE_2)
    y_idx3 = tl.arange(0, Y_SHAPE_3)
    y_idx4 = tl.arange(0, Y_SHAPE_4)

    xidx = x_idx0[:, None, None, None, None] * X_SHAPE_1 * X_SHAPE_2 * X_SHAPE_3 * X_SHAPE_4 + \
           x_idx1[None, :, None, None, None] * X_SHAPE_2 * X_SHAPE_3 * X_SHAPE_4 + \
           x_idx2[None, None, :, None, None] * X_SHAPE_3 * X_SHAPE_4 + \
           x_idx3[None, None, None, :, None] * X_SHAPE_4 + x_idx4[None, None, None, None, :]

    yidx = y_idx0[:, None, None, None, None] * Y_SHAPE_1 * Y_SHAPE_2 * Y_SHAPE_3 * Y_SHAPE_4 + \
           y_idx1[None, :, None, None, None] * Y_SHAPE_2 * Y_SHAPE_3 * Y_SHAPE_4 + \
           y_idx2[None, None, :, None, None] * Y_SHAPE_3 * Y_SHAPE_4 + \
           y_idx3[None, None, None, :, None] * Y_SHAPE_4 + y_idx4[None, None, None, None, :]

    X = tl.load(in_ptr0 + xidx)
    Y = tl.load(in_ptr1 + yidx)
    ret = X + Y

    tl.store(out_ptr0 + xidx, ret)


@triton.jit
def triton_add_4d_5d(
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
    ret = x_val + y_val
    tl.store(output_ptr + offsets, ret, mask=masks)


@pytest.mark.parametrize('shape', TestUtils.full_shape)
@pytest.mark.parametrize('dtype', ['bool', 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16'])
def test_add(shape, dtype):
    logging.log(logging.DEBUG, f"shape = {shape}")
    x = test_common.generate_tensor(shape, dtype).npu()
    y = test_common.generate_tensor(shape, dtype).npu()
    z = test_common.generate_tensor(shape, dtype).npu()

    ans = x + y
    output = torch.zeros_like(ans)

    if len(shape) == 1:
        triton_add[1, 1, shape[0]](output, x, y, z, 1, 1, 1, 1, 1, shape[0])
    elif len(shape) == 2:
        if shape[0] > shape[1]:
            triton_add[1, shape[0], 1](output, x, y, z, 1, 1, shape[1], 1, shape[0], shape[1])
        else:
            triton_add[1, 1, shape[1]](output, x, y, z, 1, shape[0], 1, 1, shape[0], shape[1])
    elif len(shape) == 3:
        if max(shape[0], shape[1], shape[2]) == shape[0]:
            triton_add[shape[0], 1, 1](output, x, y, z, 1, shape[1], shape[2], shape[0], shape[1], shape[2])
        elif max(shape[0], shape[1], shape[2]) == shape[1]:
            triton_add[1, shape[1], 1](output, x, y, z, shape[0], 1, shape[2], shape[0], shape[1], shape[2])
        else:
            triton_add[1, 1, shape[2]](output, x, y, z, shape[0], shape[1], 1, shape[0], shape[1], shape[2])
    else:
        triton_add[1, 1, 1](output, x, y, z, 1, 1, 1, 1, 1, 1)

    test_common.validate_cmp(dtype, ans, output)


@pytest.mark.parametrize('shape', TestUtils.test_shape4d + TestUtils.test_shape5d)
@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16'])
def test_add_4d_5d(shape, dtype):
    logging.log(logging.DEBUG, f"shape = {shape}")
    x = test_common.generate_tensor(shape, dtype).npu()
    y = test_common.generate_tensor(shape, dtype).npu()

    output = torch.randint(1, shape, dtype=eval('torch.' + dtype)).npu()

    logging.log(logging.DEBUG, f"output.dtype={output.dtype}")

    ans = x + y

    blocks = list(x.size())
    strides = list(x.stride())
    while len(blocks) < 5:
        blocks.append(1)
        strides.append(1)

    grid = (1,)
    triton_add_4d_5d[grid](output, x, y, *blocks, *blocks, *strides)

    test_common.validate_cmp(dtype, ans, output)


def promote_dtype(x_dtype, y_dtype):
    """
    如果 y 的精度低于 x, 则提升 y 的精度以匹配 x。
    """
    # 如果两个数据类型一致，直接返回
    if x_dtype == y_dtype:
        return y_dtype
    
    # 构建类型的优先级列表（从低到高）
    priority = [
        torch.bool, torch.int8, torch.int16, torch.int32, torch.int64,
        torch.float16, torch.bfloat16, torch.float32
    ]

    # 查找两种类型在优先级列表中的位置
    x_priority = priority.index(x_dtype)
    y_priority = priority.index(y_dtype)

    # 如果y的优先级比x小，则提升到x的类型
    if y_priority < x_priority:
        return x_dtype
    else:
        return y_dtype


@pytest.mark.parametrize('param_list',
                         [
                            [(5, 1, 1, 1, 1), (5, 1, 1, 2, 1)],
                            [(2, 1), (2, 4)],
                            [(2, 1, 1), (2, 4, 2)],
                            [(2, 1, 1, 1), (2, 4, 2, 2)],
                            [(2, 1, 1, 1, 1), (2, 4, 2, 2, 2)],
                            [(1, ), (4, )],
                            [(1, 2, 1), (1, 2, 3)],
                            [(1, 1, 1, 1), (7, 1, 1, 1)]
                         ]
                         )
@pytest.mark.parametrize('x_dtype_str', ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'bool'])
@pytest.mark.parametrize('y_dtype_str', ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'bool'])
def test_add_broadcast(param_list, x_dtype_str, y_dtype_str):
    x_shape, y_shape = param_list
    x_dtype = eval('torch.' + x_dtype_str)
    y_dtype = eval('torch.' + y_dtype_str)

    x = test_common.generate_tensor(x_shape, x_dtype_str).npu()
    y = test_common.generate_tensor(y_shape, y_dtype_str).npu()
    if y.numel() > x.numel():
        tmp = y
        y = x
        x = tmp
    ans = x + y
    while x.dim() < 5:
        x = x.unsqueeze(-1)
    while y.dim() < 5:
        y = y.unsqueeze(-1)
    bf2fpFlag = False
    out_dtype = promote_dtype(x_dtype, y_dtype)
    if (x_dtype == torch.bfloat16 and y_dtype == torch.float16) or \
       (x_dtype == torch.float16 and y_dtype == torch.bfloat16):
        out_dtype = torch.float32
        bf2fpFlag = True
    out_dtype = str(out_dtype).split('.')[-1]
    out = test_common.generate_tensor(x.shape, out_dtype).npu()

    triton_add_broadcast[1, 1, 1](x, y, out, *x.shape, *y.shape)
    while out.dim() > ans.dim():
        out = out.squeeze(-1)

    if bf2fpFlag:
        torch.testing.assert_close(out, ans, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(out, ans)
    


@triton.jit
def add_5d(x_ptr, y_ptr, out_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr, NB: tl.constexpr,
                            XB1: tl.constexpr, YB1: tl.constexpr, ZB1: tl.constexpr, MB1: tl.constexpr, NB1: tl.constexpr):
    offsets = tl.arange(0, XB) * (YB * ZB * MB * NB)
    offsets = offsets[:, None] + tl.arange(0, YB)[None, :] * (ZB * MB * NB)
    offsets = offsets[:, :, None] + tl.arange(0, ZB)[None, None, :] * (MB * NB)
    offsets = offsets[:, :, :, None] + tl.arange(0, MB)[None, None, None, :] * NB
    offsets = offsets[:, :, :, :, None] + tl.arange(0, NB)[None, None, None, None, :]

    offsets1 = tl.arange(0, XB1) * (YB1 * ZB1 * MB1 * NB1)
    offsets1 = offsets1[:, None] + tl.arange(0, YB1)[None, :] * (ZB1 * MB1 * NB1)
    offsets1 = offsets1[:, :, None] + tl.arange(0, ZB1)[None, None, :] * (MB1 * NB1)
    offsets1 = offsets1[:, :, :, None] + tl.arange(0, MB1)[None, None, None, :] * NB1
    offsets1 = offsets1[:, :, :, :, None] + tl.arange(0, NB1)[None, None, None, None, :]
    
    tmp0 = tl.load(x_ptr + offsets)
    tmp1 = tl.load(y_ptr + offsets1)
    tmp2 = tl.load(out_ptr + offsets1)
    out = tmp2 + tmp1 + tmp0
    tl.store(out_ptr + offsets1, out)


@triton.jit
def add_4d(x_ptr, y_ptr, out_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr,
                            XB1: tl.constexpr, YB1: tl.constexpr, ZB1: tl.constexpr, MB1: tl.constexpr):
    offsets = tl.arange(0, XB) * (YB * ZB * MB)
    offsets = offsets[:, None] + tl.arange(0, YB)[None, :] * (ZB * MB)
    offsets = offsets[:, :, None] + tl.arange(0, ZB)[None, None, :] * (MB)
    offsets = offsets[:, :, :, None] + tl.arange(0, MB)[None, None, None, :] 

    offsets1 = tl.arange(0, XB1) * (YB1 * ZB1 * MB1)
    offsets1 = offsets1[:, None] + tl.arange(0, YB1)[None, :] * (ZB1 * MB1)
    offsets1 = offsets1[:, :, None] + tl.arange(0, ZB1)[None, None, :] * (MB1)
    offsets1 = offsets1[:, :, :, None] + tl.arange(0, MB1)[None, None, None, :]
    
    tmp0 = tl.load(x_ptr + offsets)
    tmp1 = tl.load(y_ptr + offsets1)
    tmp2 = tl.load(out_ptr + offsets1)
    out = tmp2 + tmp1 + tmp0
    tl.store(out_ptr + offsets1, out)


@triton.jit
def add_3d(x_ptr, y_ptr, out_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
                            XB1: tl.constexpr, YB1: tl.constexpr, ZB1: tl.constexpr):
    offsets = tl.arange(0, XB) * (YB * ZB)
    offsets = offsets[:, None] + tl.arange(0, YB)[None, :] * (ZB)
    offsets = offsets[:, :, None] + tl.arange(0, ZB)[None, None, :]

    offsets1 = tl.arange(0, XB1) * (YB1 * ZB1)
    offsets1 = offsets1[:, None] + tl.arange(0, YB1)[None, :] * (ZB1)
    offsets1 = offsets1[:, :, None] + tl.arange(0, ZB1)[None, None, :]
    
    tmp0 = tl.load(x_ptr + offsets)
    tmp1 = tl.load(y_ptr + offsets1)
    tmp2 = tl.load(out_ptr + offsets1)
    out = tmp2 + tmp1 + tmp0
    tl.store(out_ptr + offsets1, out)


@triton.jit
def add_2d(x_ptr, y_ptr, out_ptr, XB: tl.constexpr, YB: tl.constexpr,
                            XB1: tl.constexpr, YB1: tl.constexpr):
    offsets = tl.arange(0, XB) * (YB)
    offsets = offsets[:, None] + tl.arange(0, YB)[None, :]

    offsets1 = tl.arange(0, XB1) * (YB1)
    offsets1 = offsets1[:, None] + tl.arange(0, YB1)[None, :]
    
    tmp0 = tl.load(x_ptr + offsets)
    tmp1 = tl.load(y_ptr + offsets1)
    tmp2 = tl.load(out_ptr + offsets1)
    out = tmp2 + tmp1 + tmp0
    tl.store(out_ptr + offsets1, out)


@pytest.mark.parametrize('param_list',
    [
        [(5, 1, 1, 1, 1), (5, 1, 1, 2, 1)],
    ]
    )
@pytest.mark.parametrize('x_dtype_str', ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'bool'])
@pytest.mark.parametrize('y_dtype_str', ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'bool'])
def test_add_2d_to_5d(x_dtype_str, y_dtype_str, param_list):
    x0_shape, y_shape = param_list
    ndim = max(len(x0_shape), len(y_shape))
    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    y_dtype = eval('torch.' + y_dtype_str)

    x0 = test_common.generate_tensor(x0_shape, x_dtype_str).npu()
    y = test_common.generate_tensor(y_shape, y_dtype_str).npu()

    out_dtype = promote_dtype(x_dtype, y_dtype)
    if out_dtype == torch.bfloat16:
        out_dtype = torch.float32
    out = torch.full(y_shape, 0, dtype=out_dtype).npu()

    x0_temp = x0.clone()
    y_temp = y.clone()
    out_temp = out.clone()

    triton_shape = [*x0_shape]
    while len(triton_shape) < ndim:
        triton_shape.append(1)

    triton_shape1 = [*y_shape]
    while len(triton_shape1) < ndim:
        triton_shape1.append(1)

    # 按维度分支
    if ndim == 2:
        XB, YB = triton_shape
        XB1, YB1 = triton_shape1

        add_2d[(1, )](
            x_ptr=x0,
            y_ptr=y,
            out_ptr=out,
            XB=XB, YB=YB,
            XB1=XB1, YB1=YB1,
        )

    elif ndim == 3:
        XB, YB, ZB = triton_shape
        XB1, YB1, ZB1 = triton_shape1

        add_3d[(1, )](
            x_ptr=x0,
            y_ptr=y,
            out_ptr=out,
            XB=XB, YB=YB, ZB=ZB,
            XB1=XB1, YB1=YB1, ZB1=ZB1,
        )

    elif ndim == 4:
        XB, YB, ZB, MB = triton_shape
        XB1, YB1, ZB1, MB1 = triton_shape1

        add_4d[(1, )](
            x_ptr=x0,
            y_ptr=y,
            out_ptr=out,
            XB=XB, YB=YB, ZB=ZB, MB=MB,
            XB1=XB1, YB1=YB1, ZB1=ZB1, MB1=MB1,
        )

    elif ndim == 5:
        XB, YB, ZB, MB, NB = triton_shape
        XB1, YB1, ZB1, MB1, NB1 = triton_shape1

        add_5d[(1, )](
            x_ptr=x0,
            y_ptr=y,
            out_ptr=out,
            XB=XB, YB=YB, ZB=ZB, MB=MB, NB=NB,
            XB1=XB1, YB1=YB1, ZB1=ZB1, MB1=MB1, NB1=NB1,
        )

    else:
        raise ValueError(f"Unsupported tensor dim: {ndim}")
    expected = out_temp + y_temp + x0_temp
    torch.testing.assert_close(out, expected)