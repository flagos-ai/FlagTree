# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
import math
import random
import pytest
import torch
import triton
import triton.language as tl

import test_common
from test_common import TestUtils, get_dtype_size


def torch_reduce(x1, dim):
    if x1.dtype == torch.float16 or x1.dtype == torch.float32:
        res = torch.sum(x1.to(torch.float32), dim=dim).to(x1.dtype)
    else:
        res = torch.sum(x1, dim=dim).to(x1.dtype)
    return res


@triton.jit
def _reduce_combine(a, b):
    return a + b


@triton.jit
def tt_reduce_1d(in_ptr, out_ptr,
                 xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                 XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, dim: tl.constexpr):
    idx = tl.arange(0, XB)
    x = tl.load(in_ptr + idx)
    ret = tl.reduce(x, dim, _reduce_combine)
    tl.store(out_ptr + tl.arange(0, 1), ret)


@triton.jit
def tt_reduce_2d(in_ptr, out_ptr,
                 xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                 XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, dim: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    idx = xidx[:, None] * ynumel + yidx[None, :]

    x = tl.load(in_ptr + idx)
    ret = tl.reduce(x, dim, _reduce_combine)

    if dim == 0:
        oidx = yidx
    else:
        oidx = xidx
    tl.store(out_ptr + oidx, ret)


@triton.jit
def tt_reduce_1d_dim_none(in_ptr, out_ptr,
                          xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                          XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, dim: tl.constexpr):
    idx = tl.arange(0, XB)
    x = tl.load(in_ptr + idx)
    ret = tl.reduce(x, dim, _reduce_combine)
    tl.store(out_ptr + tl.arange(0, 1), ret)


@triton.jit
def tt_reduce_2d_dim_none(in_ptr, out_ptr,
                          xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                          XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, dim: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    idx = xidx[:, None] * ynumel + yidx[None, :]

    x = tl.load(in_ptr + idx)
    ret = tl.reduce(x, dim, _reduce_combine)

    tl.store(out_ptr + tl.arange(0, 1), ret)


@triton.jit
def tt_reduce_3d_dim_none(in_ptr, out_ptr,
                          xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                          XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, dim: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs

    idx = xidx[:, None, None] * ynumel * znumel + yidx[None, :, None] * znumel + zidx[None, None, :]

    x = tl.load(in_ptr + idx)
    ret = tl.reduce(x, dim, _reduce_combine)

    tl.store(out_ptr, ret)


@triton.jit
def tt_reduce_3d(in_ptr, out_ptr,
                 xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                 XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, dim: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs

    idx = xidx[:, None, None] * ynumel * znumel + yidx[None, :, None] * znumel + zidx[None, None, :]

    x = tl.load(in_ptr + idx)
    ret = tl.reduce(x, dim, _reduce_combine)

    if dim == 0:
        oidx = yidx[:, None] * znumel + zidx[None, :]
    elif dim == 1:
        oidx = xidx[:, None] * znumel + zidx[None, :]
    else:
        oidx = xidx[:, None] * ynumel + yidx[None, :]

    tl.store(out_ptr + oidx, ret)


@triton.jit
def tt_reduce_3d_0_1(in_ptr, out_ptr,
                     xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                     XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, dim: tl.constexpr):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)

    idx = xidx[:, None, None] * ynumel * znumel + yidx[None, :, None] * znumel + zidx[None, None, :]

    x = tl.load(in_ptr + idx)

    tmp = tl.reduce(x, 0, _reduce_combine)
    ret = tl.reduce(tmp, 0, _reduce_combine)
    oidx = zidx

    tl.store(out_ptr + oidx, ret)


@triton.jit
def tt_reduce_3d_0_2(in_ptr, out_ptr,
                     xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                     XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, dim: tl.constexpr):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)

    idx = xidx[:, None, None] * ynumel * znumel + yidx[None, :, None] * znumel + zidx[None, None, :]

    x = tl.load(in_ptr + idx)

    tmp = tl.reduce(x, 0, _reduce_combine)
    ret = tl.reduce(tmp, 1, _reduce_combine)
    oidx = yidx

    tl.store(out_ptr + oidx, ret)


@triton.jit
def tt_reduce_3d_1_2(in_ptr, out_ptr,
                     xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                     XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, dim: tl.constexpr):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)

    idx = xidx[:, None, None] * ynumel * znumel + yidx[None, :, None] * znumel + zidx[None, None, :]

    x = tl.load(in_ptr + idx)

    tmp = tl.reduce(x, 1, _reduce_combine)
    ret = tl.reduce(tmp, 1, _reduce_combine)
    oidx = xidx

    tl.store(out_ptr + oidx, ret)


def is_legal_combine(shape, dims):
    return dims is None or (len(shape) == 3) or \
        (len(dims) == 1 and dims[0] < len(shape))


dims_map = {
    (0, 1): tt_reduce_3d_0_1,
    (1, 2): tt_reduce_3d_1_2,
    (0, 2): tt_reduce_3d_0_2
}

shape_map = {
    1: {"append_shape": (1, 1), "func": tt_reduce_1d},
    2: {"append_shape": (1,), "func": tt_reduce_2d},
    3: {"append_shape": (), "func": tt_reduce_3d}
}


def reduce_check_ub_mem_overflow(dtype, shape):
    dtype_size = get_dtype_size(dtype)
    if (dtype == "int8" or dtype == "bool") and dtype_size * math.prod(shape) >= (TestUtils.ub_size / 20):
        pytest.skip("dtype:{dtype} shape:{shape} mem overflow, skipping.")
    elif dtype_size * math.prod(shape) >= (TestUtils.ub_size / 6):
        pytest.skip("dtype:{dtype} shape:{shape} mem overflow, skipping.")


@pytest.mark.parametrize('shape', random.sample(TestUtils.full_shape, 5))
@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
@pytest.mark.parametrize('dims', [None, (0,), (1,), (2,), (0, 1), (1, 2), (0, 2)])
def test_reduce(dtype, shape, dims):
    if not is_legal_combine(shape, dims):
        return

    torch.manual_seed(0)
    x = test_common.generate_tensor(shape, dtype).npu()
    grid = (1, 1, 1)

    y_ref = torch_reduce(x, dims)
    y_cal = torch.empty(y_ref.shape, dtype=eval('torch.' + dtype), device="npu")

    if dims is None:
        reduce_check_ub_mem_overflow(dtype, shape)
        append_shape, tt_kernel = shape_map[len(shape)]["append_shape"], shape_map[len(shape)]["func"]
        xnumel, ynumel, znumel = shape + append_shape
        XB, YB, ZB = xnumel, ynumel, znumel
        if len(shape) == 1:
            tt_reduce_1d_dim_none[1, 1, 1](x, y_cal, xnumel, ynumel, znumel, XB, YB, ZB, dims)
        if len(shape) == 2:
            tt_reduce_2d_dim_none[1, 1, 1](x, y_cal, xnumel, ynumel, znumel, XB, YB, ZB, dims)
        if len(shape) == 3:
            tt_reduce_3d_dim_none[1, 1, 1](x, y_cal, xnumel, ynumel, znumel, XB, YB, ZB, dims)

        test_common.validate_cmp(dtype, y_cal, y_ref)

    elif len(dims) == 1:  # 1d reduce, 1-3d shape
        append_shape, tt_kernel = shape_map[len(shape)]["append_shape"], shape_map[len(shape)]["func"]
        xnumel, ynumel, znumel = shape + append_shape
        XB, YB, ZB = xnumel, ynumel, znumel
        if (len(shape) == 2) and (x.numel() * x.element_size() > 8192):
            if dims[0] == 0:
                grid = (1, ynumel, 1)
                YB = 1
            else:
                grid = (xnumel, 1, 1)
                XB = 1
        tt_kernel[grid](x, y_cal, xnumel, ynumel, znumel, XB, YB, ZB, dims[0])
        test_common.validate_cmp(dtype, y_cal, y_ref)
    else:  # 3d shape, 2d reduce
        tt_kernel = dims_map[dims]
        xnumel, ynumel, znumel = shape
        XB, YB, ZB = xnumel, ynumel, znumel

        tt_kernel[grid](x, y_cal, xnumel, ynumel, znumel, XB, YB, ZB, dims[0])
        test_common.validate_cmp(dtype, y_cal, y_ref)


@triton.jit
def triton_reduce_multi_d(in_ptr, out_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr,
                          NB: tl.constexpr, DIMS: tl.constexpr, DIM: tl.constexpr, REDUCE_NUMEL: tl.constexpr):
    offsets = tl.arange(0, XB) * (YB * ZB * MB * NB)
    if DIMS > 1:
        offsets = offsets[:, None] + tl.arange(0, YB)[None, :] * (ZB * MB * NB)
    if DIMS > 2:
        offsets = offsets[:, :, None] + tl.arange(0, ZB)[None, None, :] * (MB * NB)
    if DIMS > 3:
        offsets = offsets[:, :, :, None] + tl.arange(0, MB)[None, None, None, :] * NB
    if DIMS > 4:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, NB)[None, None, None, None, :]

    x = tl.load(in_ptr + offsets)

    if DIM is not None:
        ret = tl.reshape(tl.reduce(x, DIM, _reduce_combine), REDUCE_NUMEL)
        o_offsets = tl.arange(0, REDUCE_NUMEL)
        tl.store(out_ptr + o_offsets, ret)
    else:
        ret = tl.reduce(x, DIM, _reduce_combine)
        tl.store(out_ptr, ret)


@pytest.mark.shape_4d_5d
@pytest.mark.parametrize('shape', [
    (4, 2, 8, 4),
    (4, 3, 8, 1),
])
@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
@pytest.mark.parametrize('dims', [None, (0,), (1,), (2,), (3,)])
def test_reduce_4d(dtype, shape, dims):
    torch.manual_seed(0)

    x = test_common.generate_tensor(shape, dtype).npu()
    dim = dims[0] if dims is not None else None

    y_ref = torch_reduce(x, dim)
    y_cal = torch.empty(y_ref.shape, dtype=eval('torch.' + dtype), device="npu")

    triton_shape = [*shape]
    while len(triton_shape) < 5:
        triton_shape.append(1)
    reduce_numel = math.prod(triton_shape) // triton_shape[dim] if dim is not None else None
    grid = (1,)
    triton_reduce_multi_d[grid](x, y_cal, *triton_shape, len(shape), dim, reduce_numel)
    test_common.validate_cmp(dtype, y_cal, y_ref)


@pytest.mark.shape_4d_5d
@pytest.mark.parametrize('shape', [
    (2, 4, 2, 8, 4),
    (3, 4, 2, 8, 1),
])
@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
@pytest.mark.parametrize('dims', [None, (0,), (1,), (2,), (3,), (4,)])
def test_reduce_5d(dtype, shape, dims):
    torch.manual_seed(0)

    x = test_common.generate_tensor(shape, dtype).npu()
    dim = dims[0] if dims is not None else None

    y_ref = torch_reduce(x, dim)
    y_cal = torch.empty(y_ref.shape, dtype=eval('torch.' + dtype), device="npu")

    triton_shape = [*shape]
    while len(triton_shape) < 5:
        triton_shape.append(1)
    reduce_numel = math.prod(triton_shape) // triton_shape[dim] if dim is not None else None
    grid = (1,)
    triton_reduce_multi_d[grid](x, y_cal, *triton_shape, len(shape), dim, reduce_numel)
    test_common.validate_cmp(dtype, y_cal, y_ref)
