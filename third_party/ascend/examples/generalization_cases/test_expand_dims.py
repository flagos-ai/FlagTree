# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import triton
import triton.language as tl

import torch
import torch_npu
import pytest
import test_common
from test_common import TestUtils
import logging


@triton.jit
def fn_npu_1d(output_ptr, x_ptr, YB: tl.constexpr, ZB: tl.constexpr):
    yidx = tl.arange(0, YB)

    X = tl.load(x_ptr + yidx)

    ret = tl.expand_dims(X, 1)

    oidx = yidx[:, None] + tl.arange(0, 1)[None, :]

    tl.store(output_ptr + oidx, ret)


@pytest.mark.parametrize('shape', TestUtils.test_shape1d)
@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
def test_expand_dims_1d(shape, dtype):
    x = test_common.generate_tensor(shape,dtype).npu()
    a = x.unsqueeze(1)

    output = torch.randint(1, (shape[0], 1), dtype=eval('torch.' + dtype)).npu()

    fn_npu_1d[1, 1, 1](output, x, YB=shape[0], ZB=1, debug=True)

    torch.testing.assert_close(output, a)


@triton.jit
def fn_npu_2d(output_ptr, x_ptr, YB: tl.constexpr, ZB: tl.constexpr):
    yoffs = tl.program_id(0)
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB)

    idx = yidx[:, None] * ZB + zidx[None, :]

    X = tl.load(x_ptr + idx)

    ret = tl.expand_dims(X, 1)

    oidx = yidx[:, None, None] * ZB + tl.arange(0, 1)[None, :, None] + zidx[None, None, :]

    tl.store(output_ptr + oidx, ret)


@pytest.mark.parametrize('shape', TestUtils.test_shape2d)
@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
def test_expand_dims_2d(shape, dtype):
    x = test_common.generate_tensor(shape,dtype).npu()
    a = x.unsqueeze(1)

    output = torch.randint(1, (shape[0], 1, shape[1]), dtype=eval('torch.' + dtype)).npu()

    if x.numel()*x.element_size()>8192:
        fn_npu_2d[shape[0],1 ,1](output, x, YB=1, ZB=shape[1])
    else:
        fn_npu_2d[1, 1, 1](output, x, YB=shape[0], ZB=shape[1])

    torch.testing.assert_close(output, a)


@triton.jit
def fn_npu_3d(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)

    idx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    X = tl.load(x_ptr + idx)

    ret = tl.expand_dims(X, 2)

    oidx = xidx[:, None, None, None] * YB * ZB + yidx[None, :, None, None] * ZB + tl.arange(0, 1)[None, None, :,
                                                                                  None] + zidx[None, None, None, :]

    tl.store(output_ptr + oidx, ret)


@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
@pytest.mark.parametrize('shape', TestUtils.test_shape3d)
def test_expand_dims_3d(dtype, shape):
    x = test_common.generate_tensor(shape,dtype).npu()
    a = x.unsqueeze(2)

    output = torch.randint(1, (shape[0], shape[1], 1, shape[2]), dtype=eval('torch.' + dtype)).npu()

    fn_npu_3d[1, 1, 1](output, x, XB=shape[0], YB=shape[1], ZB=shape[2])

    torch.testing.assert_close(output, a)


@triton.jit
def fn_npu_multi_d(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr, NB: tl.constexpr, DIMS: tl.constexpr, DIM: tl.constexpr):
    in_offsets = tl.arange(0, XB) * (YB * ZB * MB * NB)
    if DIMS > 1:
        in_offsets = in_offsets[:, None] + tl.arange(0, YB)[None, :] * (ZB * MB * NB)
    if DIMS > 2:
        in_offsets = in_offsets[:, :, None] + tl.arange(0, ZB)[None, None, :] * (MB * NB)
    if DIMS > 3:
        in_offsets = in_offsets[:, :, :, None] + tl.arange(0, MB)[None, None, None, :] * NB
    if DIMS > 4:
        in_offsets = in_offsets[:, :, :, :, None] + tl.arange(0, NB)[None, None, None, None, :]

    X = tl.load(x_ptr + in_offsets)

    ret = tl.expand_dims(X, DIM).reshape(XB * YB * ZB * MB * NB)

    out_offstes = tl.arange(0, XB * YB * ZB * MB * NB)
    tl.store(output_ptr + out_offstes, ret)


@pytest.mark.shape_4d_5d
@pytest.mark.parametrize('dtype', ['int8', 'float16', 'float32'])
@pytest.mark.parametrize('shape', [
    (2, 64, 16, 2),
    (8, 8, 4, 2),
    (8, 8, 4, 1),
])
@pytest.mark.parametrize('dim', [-1, 0, 1, 2, 3])
def test_npu_4d(shape, dtype, dim):
    x = test_common.generate_tensor(shape, dtype).npu()
    expected = x.unsqueeze(dim)

    output = torch.empty_like(expected)

    triton_shape = [*shape]
    while len(triton_shape) < 5:
        triton_shape.append(1)
    grid = (1, )
    fn_npu_multi_d[grid](output, x, *triton_shape, len(shape), dim)

    torch.testing.assert_close(output, expected)


@pytest.mark.shape_4d_5d
@pytest.mark.parametrize('dtype', ['int8', 'float16', 'float32'])
@pytest.mark.parametrize('shape', [
    (2, 32, 3, 16, 2),
    (8, 8, 3, 4, 2),
    (8, 8, 3, 4, 1),
])
@pytest.mark.parametrize('dim', [-1, 0, 1, 2, 3, 4])
def test_npu_5d(shape, dtype, dim):
    x = test_common.generate_tensor(shape, dtype).npu()
    expected = x.unsqueeze(dim)

    output = torch.empty_like(expected)

    triton_shape = [*shape]
    while len(triton_shape) < 5:
        triton_shape.append(1)
    grid = (1, )
    fn_npu_multi_d[grid](output, x, *triton_shape, len(shape), dim)

    torch.testing.assert_close(output, expected)
