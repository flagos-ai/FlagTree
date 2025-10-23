# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
import logging
import math
import pytest
import torch
import torch_npu
import numpy as np
import triton
import triton.language as tl

import test_common
from test_common import TestUtils, check_ub_mem_overflow, get_dtype_size

logger = logging.getLogger(__name__)


# <<<<<<< test_argmax_1d
def torch_argmax(x0, dim, keepdim):
    x0 = x0 if x0.device == "cpu" else x0.cpu()
    return torch.argmax(x0, dim=dim, keepdim=keepdim).npu()


@triton.jit
def triton_argmax_1d(in_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) + tl.arange(0, XBLOCK)
    tmp0 = tl.load(in_ptr0 + xoffset, None)
    tmp4 = tl.argmax(tmp0, 0)
    tl.store(out_ptr1, tmp4, None)


@pytest.mark.parametrize('shape', TestUtils.test_shape1d)
@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64', 'float16', 'bfloat16', 'float32'])
def test_argmax_1d(dtype, shape):
    dtype_size = get_dtype_size(dtype)
    if dtype_size * math.prod(shape) >= (TestUtils.ub_size / 3):
        logger.warning(f"dtype:{dtype} shape:{shape} mem overflow")
        return
    x0 = test_common.generate_tensor(shape, dtype).npu()
    triton_res = torch.empty(1, dtype=torch.int32).npu()
    numel = shape[0]
    triton_argmax_1d[1, 1, 1](x0, triton_res, numel, numel)
    torch_res = torch_argmax(x0, dim=0, keepdim=True)
    test_common.validate_cmp("int32", triton_res, torch_res)


# >>>>>>> test_argmax_1d

# <<<<<<< test_argmax_2d
@triton.jit
def triton_argmax_2d(in_ptr0, out_ptr0, dim : tl.constexpr, M : tl.constexpr, N : tl.constexpr, MNUMEL: tl.constexpr, NNUMEL: tl.constexpr):
    mblk_idx = tl.arange(0, MNUMEL)
    nblk_idx = tl.arange(0, NNUMEL)
    mmask = mblk_idx < M
    nmask = nblk_idx < N
    mask = (mmask[:, None]) & (nmask[None, :])
    idx = mblk_idx[:, None] * N + nblk_idx[None, :]
    x = tl.load(in_ptr0 + idx, mask=mask, other=-float('inf'))
    tmp4 = tl.argmax(x, dim)
    if dim == 0:
        tl.store(out_ptr0 + tl.arange(0, N), tmp4, None)
    else:
        tl.store(out_ptr0 + tl.arange(0, M), tmp4, None)


@pytest.mark.parametrize('shape', TestUtils.test_shape2d)
@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64', 'float16', 'bfloat16', 'float32'])
@pytest.mark.parametrize('dim', [0, 1])
def test_argmax_2d(dtype, shape, dim):
    dtype_size = get_dtype_size(dtype)
    if dtype_size * math.prod(shape) >= (TestUtils.ub_size / 3):
        logger.warning(f"dtype:{dtype} shape:{shape} mem overflow")
        return
    shapex, shapey = shape
    x0 = test_common.generate_tensor(shape, dtype).npu()
    triton_res = torch.empty([shape[1 - dim], ], dtype=torch.int32).npu()
    triton_argmax_2d[1, 1, 1](x0, triton_res, dim, shapex, shapey, shapex, shapey)
    torch_res = torch_argmax(x0, dim=dim, keepdim=False)
    test_common.validate_cmp("int32", triton_res, torch_res)


# >>>>>>> test_argmax_2d

# <<<<<<< test_argmax_3d
def torch_argmax_3d(x0, no_reduce_dim):
    x0 = x0 if x0.device == "cpu" else x0.cpu()
    if x0.dtype in (torch.int8, torch.int16, torch.int32):
        x0 = x0.to(torch.int64)
    if no_reduce_dim == 0:
        return torch.argmax(torch.max(x0, 1)[0], 1).npu()
    elif no_reduce_dim == 1:
        return torch.argmax(torch.max(x0, 0)[0], 1).npu()
    elif no_reduce_dim == 2:
        return torch.argmax(torch.max(x0, 0)[0], 0).npu()
    else:
        assert False, f"no reduce dim not right, no_reduce_dim = {no_reduce_dim}"


@triton.jit
def triton_argmax_3d_0_1(in_ptr, out_ptr,
                         xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                         XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)

    idx = xidx[:, None, None] * ynumel * znumel + yidx[None, :, None] * znumel + zidx[None, None, :]

    x = tl.load(in_ptr + idx)

    tmp = tl.max(x, 0)
    ret = tl.argmax(tmp, 0)
    oidx = zidx
    tl.store(out_ptr + oidx, ret)


@triton.jit
def triton_argmax_3d_0_2(in_ptr, out_ptr,
                         xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                         XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)

    idx = xidx[:, None, None] * ynumel * znumel + yidx[None, :, None] * znumel + zidx[None, None, :]

    x = tl.load(in_ptr + idx)

    tmp = tl.max(x, 0)
    ret = tl.argmax(tmp, 1)
    oidx = yidx
    tl.store(out_ptr + oidx, ret)


@triton.jit
def triton_argmax_3d_1_2(in_ptr, out_ptr,
                         xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                         XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)

    idx = xidx[:, None, None] * ynumel * znumel + yidx[None, :, None] * znumel + zidx[None, None, :]

    x = tl.load(in_ptr + idx)

    tmp = tl.max(x, 1)
    ret = tl.argmax(tmp, 1)
    oidx = xidx
    tl.store(out_ptr + oidx, ret)


def triton_argmax_3d(in_ptr, out_ptr, xnumel, ynumel, znumel, XB, YB, ZB, no_reduce_dim):
    if no_reduce_dim == 0:
        triton_argmax_3d_1_2[1, 1, 1](in_ptr, out_ptr, xnumel, ynumel, znumel, XB, YB, ZB)
    elif no_reduce_dim == 1:
        triton_argmax_3d_0_2[1, 1, 1](in_ptr, out_ptr, xnumel, ynumel, znumel, XB, YB, ZB)
    elif no_reduce_dim == 2:
        triton_argmax_3d_0_1[1, 1, 1](in_ptr, out_ptr, xnumel, ynumel, znumel, XB, YB, ZB)


@pytest.mark.parametrize('shape', TestUtils.test_shape3d)
@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64', 'float16', 'bfloat16', 'float32'])
@pytest.mark.parametrize('no_reduce_dim', [0, 1, 2])
def test_argmax_3d(dtype, shape, no_reduce_dim):
    x0 = test_common.generate_tensor(shape, dtype).npu()
    triton_res = torch.empty([shape[no_reduce_dim], ], dtype=torch.int32).npu()
    triton_argmax_3d(x0, triton_res, shape[0], shape[1], shape[2], shape[0], shape[1], shape[2], no_reduce_dim)
    torch_res = torch_argmax_3d(x0, no_reduce_dim)
    test_common.validate_cmp("int32", triton_res, torch_res)

# >>>>>>> test_argmax_3d


# <<<<<<< test_argmax_4d
def torch_argmax_4d(x0, dim):
    x0 = x0 if x0.device == "cpu" else x0.cpu()
    if x0.dtype in (torch.int8, torch.int16, torch.int32):
        x0 = x0.to(torch.int64)
    return torch.argmax(x0, dim)


@triton.jit
def argmax_4d(out_ptr, x, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr, DIM: tl.constexpr):
    if DIM == 0:
        ret = tl.reshape(tl.argmax(x, DIM), XB * YB * ZB * MB // XB)
        o_idx = tl.arange(0, XB * YB * ZB * MB // XB)
        tl.store(out_ptr + o_idx, ret)
    elif DIM == 1:
        ret = tl.reshape(tl.argmax(x, DIM), XB * YB * ZB * MB // YB)
        o_idx = tl.arange(0, XB * YB * ZB * MB // YB)
        tl.store(out_ptr + o_idx, ret)
    elif DIM == 2:
        ret = tl.reshape(tl.argmax(x, DIM), XB * YB * ZB * MB // ZB)
        o_idx = tl.arange(0, XB * YB * ZB * MB // ZB)
        tl.store(out_ptr + o_idx, ret)
    else:
        ret = tl.reshape(tl.argmax(x, DIM), XB * YB * ZB * MB // MB)
        o_idx = tl.arange(0, XB * YB * ZB * MB // MB)
        tl.store(out_ptr + o_idx, ret)


@triton.jit
def triton_argmax_kernel_4d(in_ptr, out_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr, DIM: tl.constexpr):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)
    midx = tl.arange(0, MB)

    idx = xidx[:, None, None, None] * YB * ZB * MB + yidx[None, :, None, None] * ZB * MB + zidx[None, None, :, None] * MB + midx[None, None, None, :]

    x = tl.load(in_ptr + idx)

    argmax_4d(out_ptr, x, XB, YB, ZB, MB, DIM)


def triton_argmax_4d(in_ptr, out_ptr, XB, YB, ZB, MB, dim):
    triton_argmax_kernel_4d[(1,)](in_ptr, out_ptr, XB, YB, ZB, MB, dim)


@pytest.mark.shape_4d_5d
@pytest.mark.parametrize('shape', [
    (2, 2, 4, 8),
    (2, 3, 4, 8),
])
@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64', 'float16', 'bfloat16', 'float32'])
@pytest.mark.parametrize('dim', [0])
def test_argmax_4d(dtype, shape, dim):
    x0 = test_common.generate_tensor(shape, dtype).npu()
    torch_res = torch_argmax_4d(x0, dim).to(torch.int32)
    triton_res = torch.empty_like(torch_res, dtype=torch.int32).npu()
    triton_argmax_4d(x0, triton_res, shape[0], shape[1], shape[2], shape[3], dim)

    test_common.validate_cmp("int32", triton_res, torch_res)
# >>>>>>> test_argmax_4d


# <<<<<<< test_argmax_5d
def torch_argmax_5d(x0, dim):
    x0 = x0 if x0.device == "cpu" else x0.cpu()
    if x0.dtype in (torch.int8, torch.int16, torch.int32):
        x0 = x0.to(torch.int64)
    return torch.argmax(x0, dim)


@triton.jit
def argmax_5d(out_ptr, x, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr, NB: tl.constexpr, DIM: tl.constexpr):
    if DIM == 0:
        ret = tl.reshape(tl.argmax(x, DIM), XB * YB * ZB * MB * NB // XB)
        o_idx = tl.arange(0, XB * YB * ZB * MB * NB // XB)
        tl.store(out_ptr + o_idx, ret)
    elif DIM == 1:
        ret = tl.reshape(tl.argmax(x, DIM), XB * YB * ZB * MB * NB // YB)
        o_idx = tl.arange(0, XB * YB * ZB * MB * NB // YB)
        tl.store(out_ptr + o_idx, ret)
    elif DIM == 2:
        ret = tl.reshape(tl.argmax(x, DIM), XB * YB * ZB * MB * NB // ZB)
        o_idx = tl.arange(0, XB * YB * ZB * MB * NB // ZB)
        tl.store(out_ptr + o_idx, ret)
    elif DIM == 3:
        ret = tl.reshape(tl.argmax(x, DIM), XB * YB * ZB * MB * NB // MB)
        o_idx = tl.arange(0, XB * YB * ZB * MB * NB // MB)
        tl.store(out_ptr + o_idx, ret)
    else:
        ret = tl.reshape(tl.argmax(x, DIM), XB * YB * ZB * MB * NB // NB)
        o_idx = tl.arange(0, XB * YB * ZB * MB * NB // NB)
        tl.store(out_ptr + o_idx, ret)


@triton.jit
def triton_argmax_kernel_5d(in_ptr, out_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr, NB: tl.constexpr, DIM: tl.constexpr):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)
    midx = tl.arange(0, MB)
    nidx = tl.arange(0, NB)

    idx = xidx[:, None, None, None, None] * YB * ZB * MB * NB + yidx[None, :, None, None, None] * ZB * MB * NB + zidx[None, None, :, None, None] * MB * NB + midx[None, None, None, :, None] * NB + nidx[None, None, None, None, :] 

    x = tl.load(in_ptr + idx)

    argmax_5d(out_ptr, x, XB, YB, ZB, MB, NB, DIM)


def triton_argmax_5d(in_ptr, out_ptr, XB, YB, ZB, MB, NB, dim):
    triton_argmax_kernel_5d[(1,)](in_ptr, out_ptr, XB, YB, ZB, MB, NB, dim)


@pytest.mark.shape_4d_5d
@pytest.mark.parametrize('shape', [
    (2, 2, 2, 4, 8),
    (2, 2, 3, 4, 8),
])
@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64', 'float16', 'bfloat16', 'float32'])
@pytest.mark.parametrize('dim', [0])
def test_argmax_5d(dtype, shape, dim):
    x0 = test_common.generate_tensor(shape, dtype).npu()
    torch_res = torch_argmax_5d(x0, dim).to(torch.int32)
    triton_res = torch.empty_like(torch_res, dtype=torch.int32).npu()
    triton_argmax_5d(x0, triton_res, shape[0], shape[1], shape[2], shape[3], shape[4], dim)

    test_common.validate_cmp("int32", triton_res, torch_res)
# >>>>>>> test_argmax_5d


# <<<<<<< test_argmax_1d_bool
@triton.jit
def triton_argmax_1d_bool(in_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) + tl.arange(0, XBLOCK)
    tmp0 = tl.load(in_ptr0 + xoffset, None).to(tl.int1)
    tmp4 = tl.argmax(tmp0, 0)
    tl.store(out_ptr1, tmp4, None)


@pytest.mark.parametrize('shape', TestUtils.test_shape1d)
@pytest.mark.parametrize('dtype', ['bool'])
def test_argmax_1d_bool(dtype, shape):
    dtype_size = get_dtype_size(dtype)
    if dtype_size * math.prod(shape) >= (TestUtils.ub_size / 3):
        logger.warning(f"dtype:{dtype} shape:{shape} mem overflow")
        return
    x0 = test_common.generate_tensor(shape, dtype)
    triton_res = torch.empty(1, dtype=torch.int32).npu()
    numel = shape[0]
    triton_argmax_1d_bool[1, 1, 1](x0.npu(), triton_res, numel, numel)
    np_res = np.argmax(x0.numpy())
    np.equal(triton_res.item(), np_res)
# >>>>>>> test_argmax_1d_bool