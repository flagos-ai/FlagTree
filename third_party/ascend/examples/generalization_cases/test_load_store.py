# -*- coding: utf-8 -*-
# # Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import triton
import triton.language as tl
import torch
import torch_npu
import pytest
import test_common
from test_common import TestUtils
import logging


@triton.jit
def fn_npu_1d(output_ptr, x_ptr, YB: tl.constexpr):
    idx = tl.arange(0, YB)
    X = tl.load(x_ptr + idx)
    tl.store(output_ptr + idx, X)

def torch_fn_npu_1d(x):
    return x

@triton.jit
def fn_npu_2d(output_ptr, x_ptr, YB: tl.constexpr, ZB: tl.constexpr):
    pid = tl.program_id(0)
    y_idx = tl.arange(0, YB)[:, None] + pid * YB
    z_idx = tl.arange(0, ZB)[None, :]
    idx = y_idx * ZB + z_idx

    X = tl.load(x_ptr + idx)

    tl.store(output_ptr + idx, X)

def torch_fn_npu_2d(x):
    return x


@triton.jit
def fn_npu_3d(output_ptr, x_ptr, YB: tl.constexpr, ZB: tl.constexpr, KB: tl.constexpr):
    y = tl.arange(0, YB)[:, None, None]
    z = tl.arange(0, ZB)[None, :, None]
    k = tl.arange(0, KB)[None, None, :]

    idx = y * ZB * KB + z * KB + k

    X = tl.load(x_ptr + idx)

    tl.store(output_ptr + idx, X)

def torch_fn_npu_3d(x):
    return x

@pytest.mark.parametrize('shape', TestUtils.test_shape1_2_3d)
@pytest.mark.parametrize('dtype', TestUtils.dtype_list)
def test_npu(shape, dtype):
    logging.debug(f'dtype:{dtype} shape:{shape}')
    data_type = eval('torch.' + dtype)
    x = torch.randint(low=0, high=2, size=shape, dtype=data_type).npu()
    triton_res = torch.empty(shape, dtype=data_type).npu()
    torch_res = x
    if len(shape) == 1:
        torch_res = torch_fn_npu_1d(x)
        fn_npu_1d[1, 1, 1](triton_res, x, shape[0])
        # uint32 转成 float32算精度，因为torch_npu不支持uint32类型张量的slice
        torch_res = torch_res if dtype != 'uint32' else torch_res.to(torch.float32)
        triton_res = triton_res if dtype != 'uint32' else triton_res.to(torch.float32)
        cmp_type = dtype if dtype != 'uint32' else 'float32'
        test_common.validate_cmp(cmp_type, triton_res[:2 * shape[0] // 3], torch_res[:2 * shape[0] // 3])
    elif len(shape) == 2:
        torch_res = torch_fn_npu_2d(x)
        fn_npu_2d[shape[0], 1, 1](triton_res, x, 1, shape[1])
        torch_res = torch_res if dtype != 'uint32' else torch_res.to(torch.float32)
        triton_res = triton_res if dtype != 'uint32' else triton_res.to(torch.float32)
        cmp_type = dtype if dtype != 'uint32' else 'float32'
        test_common.validate_cmp(cmp_type, triton_res[:2 * shape[0] // 3, :2 * shape[1] // 3],
                                 torch_res[:2 * shape[0] // 3, :2 * shape[1] // 3])
    elif len(shape) == 3:
        torch_res = torch_fn_npu_3d(x)
        fn_npu_3d[1, 1, 1](triton_res, x, shape[0], shape[1], shape[2])
        torch_res = torch_res if dtype != 'uint32' else torch_res.to(torch.float32)
        triton_res = triton_res if dtype != 'uint32' else triton_res.to(torch.float32)
        cmp_type = dtype if dtype != 'uint32' else 'float32'
        test_common.validate_cmp(cmp_type, triton_res[:2 * shape[0] // 3, :2 * shape[1] // 3, :2 * shape[2] // 3],
                                 torch_res[:2 * shape[0] // 3, :2 * shape[1] // 3, :2 * shape[2] // 3])


# require: all data (4d and 5d) can be placed into but without ub overflow
@triton.jit
def triton_load_store_multi_d(
    in_ptr0, out_ptr0, 
    BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr, BLOCK_3: tl.constexpr, BLOCK_4: tl.constexpr,
    SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr, SHAPE_3: tl.constexpr, SHAPE_4: tl.constexpr,
    STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr, STRIDE_3: tl.constexpr, STRIDE_4: tl.constexpr
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

    tmp_in = tl.load(in_ptr0 + offsets, masks)
    tmp_out = tmp_in
    tl.store(out_ptr0 + offsets, tmp_out, masks)


@pytest.mark.shape_4d_5d
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (8, 4, 16, 16)],
                             ['float16', (8, 4, 16, 16)],
                             ['int8', (8, 4, 16, 16)],
                             ['float32', (8, 8, 4, 4)],
                             ['float16', (8, 8, 4, 4)],
                             ['int8', (8, 8, 4, 4)],
                             ['float32', (3, 8, 2, 16, 16)],
                             ['float16', (3, 8, 2, 16, 16)],
                             ['int8', (9, 8, 8, 16, 16)],
                             ['float32', (11, 8, 8, 4, 4)],
                             ['float16', (11, 8, 8, 4, 4)],
                             ['int8', (11, 8, 8, 4, 4)],
                         ]
                         )
def test_load_store_4d_5d(param_list):
    # 生成数据
    dtype, shape = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    # torch结果
    y_expect = x0
    y_actual = test_common.generate_tensor(shape, dtype).npu()
    # triton结果
    blocks = list(x0.size())
    shapes = list(x0.stride())
    while len(blocks) < 5:
        blocks.append(1)
        shapes.append(1)
    triton_load_store_multi_d[(1, )](x0, y_actual, *blocks, *blocks, *shapes)
    # 比较结果
    test_common.validate_cmp(dtype, y_actual, y_expect)
