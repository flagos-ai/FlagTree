# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import math
import pytest
import random
import torch
import torch_npu
import triton
import triton.language as tl

import test_common
from test_common import TestUtils, check_ub_mem_overflow


@triton.jit
def fn_npu_int8_3d(output_ptr, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr, XNUMEL: tl.constexpr,
                   YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xidx = tl.arange(0, XNUMEL)
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Xmask = xidx < X
    Ymask = yidx < Y
    Zmask = zidx < Z
    mask = (Xmask[:, None, None]) & (Ymask[None, :, None]) & (Zmask[None, None, :])
    ret = tl.zeros((XNUMEL, YNUMEL, ZNUMEL), dtype=tl.int8)
    oidx = xidx[:, None, None] * Y * Z + yidx[None, :, None] * Z + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_int16_3d(output_ptr, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr, XNUMEL: tl.constexpr,
                    YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xidx = tl.arange(0, XNUMEL)
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Xmask = xidx < X
    Ymask = yidx < Y
    Zmask = zidx < Z
    mask = (Xmask[:, None, None]) & (Ymask[None, :, None]) & (Zmask[None, None, :])
    ret = tl.zeros((XNUMEL, YNUMEL, ZNUMEL), dtype=tl.int16)
    oidx = xidx[:, None, None] * Y * Z + yidx[None, :, None] * Z + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_int32_3d(output_ptr, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr, XNUMEL: tl.constexpr,
                    YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xidx = tl.arange(0, XNUMEL)
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Xmask = xidx < X
    Ymask = yidx < Y
    Zmask = zidx < Z
    mask = (Xmask[:, None, None]) & (Ymask[None, :, None]) & (Zmask[None, None, :])
    ret = tl.zeros((XNUMEL, YNUMEL, ZNUMEL), dtype=tl.int32)
    oidx = xidx[:, None, None] * Y * Z + yidx[None, :, None] * Z + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_int64_3d(output_ptr, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr, XNUMEL: tl.constexpr,
                    YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xidx = tl.arange(0, XNUMEL)
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Xmask = xidx < X
    Ymask = yidx < Y
    Zmask = zidx < Z
    mask = (Xmask[:, None, None]) & (Ymask[None, :, None]) & (Zmask[None, None, :])
    ret = tl.zeros((XNUMEL, YNUMEL, ZNUMEL), dtype=tl.int64)
    oidx = xidx[:, None, None] * Y * Z + yidx[None, :, None] * Z + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_fp16_3d(output_ptr, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr, XNUMEL: tl.constexpr,
                   YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xidx = tl.arange(0, XNUMEL)
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Xmask = xidx < X
    Ymask = yidx < Y
    Zmask = zidx < Z
    mask = (Xmask[:, None, None]) & (Ymask[None, :, None]) & (Zmask[None, None, :])
    ret = tl.zeros((XNUMEL, YNUMEL, ZNUMEL), dtype=tl.float16)
    oidx = xidx[:, None, None] * Y * Z + yidx[None, :, None] * Z + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_fp32_3d(output_ptr, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr, XNUMEL: tl.constexpr,
                   YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xidx = tl.arange(0, XNUMEL)
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Xmask = xidx < X
    Ymask = yidx < Y
    Zmask = zidx < Z
    mask = (Xmask[:, None, None]) & (Ymask[None, :, None]) & (Zmask[None, None, :])
    ret = tl.zeros((XNUMEL, YNUMEL, ZNUMEL), dtype=tl.float32)
    oidx = xidx[:, None, None] * Y * Z + yidx[None, :, None] * Z + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_bf16_3d(output_ptr, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr, XNUMEL: tl.constexpr,
                   YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xidx = tl.arange(0, XNUMEL)
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Xmask = xidx < X
    Ymask = yidx < Y
    Zmask = zidx < Z
    mask = (Xmask[:, None, None]) & (Ymask[None, :, None]) & (Zmask[None, None, :])
    ret = tl.zeros((XNUMEL, YNUMEL, ZNUMEL), dtype=tl.bfloat16)
    oidx = xidx[:, None, None] * Y * Z + yidx[None, :, None] * Z + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_bool_3d(output_ptr, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr, XNUMEL: tl.constexpr,
                   YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xidx = tl.arange(0, XNUMEL)
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Xmask = xidx < X
    Ymask = yidx < Y
    Zmask = zidx < Z
    mask = (Xmask[:, None, None]) & (Ymask[None, :, None]) & (Zmask[None, None, :])
    ret = tl.zeros((XNUMEL, YNUMEL, ZNUMEL), dtype=tl.int1)
    oidx = xidx[:, None, None] * Y * Z + yidx[None, :, None] * Z + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_int8_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr,
                   YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Ymask = yidx < Y
    Zmask = zidx < Z
    mask = (Ymask[:, None]) & (Zmask[None, :])
    ret = tl.zeros((YNUMEL, ZNUMEL), dtype=tl.int8)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_int16_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr,
                    YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Ymask = yidx < Y
    Zmask = zidx < Z
    mask = (Ymask[:, None]) & (Zmask[None, :])
    ret = tl.zeros((YNUMEL, ZNUMEL), dtype=tl.int16)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_int32_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr,
                    YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Ymask = yidx < Y
    Zmask = zidx < Z
    mask = (Ymask[:, None]) & (Zmask[None, :])
    ret = tl.zeros((YNUMEL, ZNUMEL), dtype=tl.int32)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_int64_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr,
                    YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Ymask = yidx < Y
    Zmask = zidx < Z
    mask = (Ymask[:, None]) & (Zmask[None, :])
    ret = tl.zeros((YNUMEL, ZNUMEL), dtype=tl.int64)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_fp16_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr,
                   YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Ymask = yidx < Y
    Zmask = zidx < Z
    mask = (Ymask[:, None]) & (Zmask[None, :])
    ret = tl.zeros((YNUMEL, ZNUMEL), dtype=tl.float16)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_fp32_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr,
                   YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Ymask = yidx < Y
    Zmask = zidx < Z
    mask = (Ymask[:, None]) & (Zmask[None, :])
    ret = tl.zeros((YNUMEL, ZNUMEL), dtype=tl.float32)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_bf16_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr,
                   YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Ymask = yidx < Y
    Zmask = zidx < Z
    mask = (Ymask[:, None]) & (Zmask[None, :])
    ret = tl.zeros((YNUMEL, ZNUMEL), dtype=tl.bfloat16)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_bool_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr,
                   YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Ymask = yidx < Y
    Zmask = zidx < Z
    mask = (Ymask[:, None]) & (Zmask[None, :])
    ret = tl.zeros((YNUMEL, ZNUMEL), dtype=tl.int1)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_int8_1d(output_ptr, Z: tl.constexpr, ZNUMEL: tl.constexpr):
    zidx = tl.arange(0, ZNUMEL)
    Zmask = zidx < Z
    mask = (Zmask[:])
    ret = tl.zeros((ZNUMEL,), dtype=tl.int8)
    oidx = zidx[:]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_int16_1d(output_ptr, Z: tl.constexpr, ZNUMEL: tl.constexpr):
    zidx = tl.arange(0, ZNUMEL)
    Zmask = zidx < Z
    mask = (Zmask[:])
    ret = tl.zeros((ZNUMEL,), dtype=tl.int16)
    oidx = zidx[:]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_int32_1d(output_ptr, Z: tl.constexpr, ZNUMEL: tl.constexpr):
    zidx = tl.arange(0, ZNUMEL)
    Zmask = zidx < Z
    mask = (Zmask[:])
    ret = tl.zeros((ZNUMEL,), dtype=tl.int32)
    oidx = zidx[:]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_int64_1d(output_ptr, Z: tl.constexpr, ZNUMEL: tl.constexpr):
    zidx = tl.arange(0, ZNUMEL)
    Zmask = zidx < Z
    mask = (Zmask[:])
    ret = tl.zeros((ZNUMEL,), dtype=tl.int64)
    oidx = zidx[:]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_fp16_1d(output_ptr, Z: tl.constexpr, ZNUMEL: tl.constexpr):
    zidx = tl.arange(0, ZNUMEL)
    Zmask = zidx < Z
    mask = (Zmask[:])
    ret = tl.zeros((ZNUMEL,), dtype=tl.float16)
    oidx = zidx[:]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_fp32_1d(output_ptr, Z: tl.constexpr, ZNUMEL: tl.constexpr):
    zidx = tl.arange(0, ZNUMEL)
    Zmask = zidx < Z
    mask = (Zmask[:])
    ret = tl.zeros((ZNUMEL,), dtype=tl.float32)
    oidx = zidx[:]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_bf16_1d(output_ptr, Z: tl.constexpr, ZNUMEL: tl.constexpr):
    zidx = tl.arange(0, ZNUMEL)
    Zmask = zidx < Z
    mask = (Zmask[:])
    ret = tl.zeros((ZNUMEL,), dtype=tl.bfloat16)
    oidx = zidx[:]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_bool_1d(output_ptr, Z: tl.constexpr, ZNUMEL: tl.constexpr):
    zidx = tl.arange(0, ZNUMEL)
    Zmask = zidx < Z
    mask = (Zmask[:])
    ret = tl.zeros((ZNUMEL,), dtype=tl.int1)
    oidx = zidx[:]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_int8_0d(output_ptr, N: tl.constexpr):
    zero = tl.zeros((), dtype=tl.int8)
    tl.store(output_ptr, zero)


@triton.jit
def fn_npu_int16_0d(output_ptr, N: tl.constexpr):
    zero = tl.zeros((), dtype=tl.int16)
    tl.store(output_ptr, zero)


@triton.jit
def fn_npu_int32_0d(output_ptr, N: tl.constexpr):
    zero = tl.zeros((), dtype=tl.int32)
    tl.store(output_ptr, zero)


@triton.jit
def fn_npu_int64_0d(output_ptr, N: tl.constexpr):
    zero = tl.zeros((), dtype=tl.int64)
    tl.store(output_ptr, zero)


@triton.jit
def fn_npu_fp16_0d(output_ptr, N: tl.constexpr):
    zero = tl.zeros((), dtype=tl.float16)
    tl.store(output_ptr, zero)


@triton.jit
def fn_npu_fp32_0d(output_ptr, N: tl.constexpr):
    zero = tl.zeros((), dtype=tl.float32)
    tl.store(output_ptr, zero)


@triton.jit
def fn_npu_bf16_0d(output_ptr, N: tl.constexpr):
    zero = tl.zeros((), dtype=tl.bfloat16)
    tl.store(output_ptr, zero)


@triton.jit
def fn_npu_bool_0d(output_ptr, N: tl.constexpr):
    zero = tl.zeros((), dtype=tl.int1)
    tl.store(output_ptr, zero)


test_dtype = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'bool']
test_shape0d = [()]
test_shape1d = TestUtils.test_shape1d
test_shape2d = TestUtils.test_shape2d
test_shape3d = TestUtils.test_shape3d

# 定义 dtype 到 (test_func, test_sigtype) 的映射
dtype_mapping3d = {
    'int8': (fn_npu_int8_3d, torch.int8),
    'int16': (fn_npu_int16_3d, torch.int16),
    'int32': (fn_npu_int32_3d, torch.int32),
    'int64': (fn_npu_int64_3d, torch.int64),
    'float16': (fn_npu_fp16_3d, torch.float16),
    'float32': (fn_npu_fp32_3d, torch.float32),
    'bfloat16': (fn_npu_bf16_3d, torch.bfloat16),
    'bool': (fn_npu_bool_3d, torch.bool),
}
dtype_mapping2d = {
    'int8': (fn_npu_int8_2d, torch.int8),
    'int16': (fn_npu_int16_2d, torch.int16),
    'int32': (fn_npu_int32_2d, torch.int32),
    'int64': (fn_npu_int64_2d, torch.int64),
    'float16': (fn_npu_fp16_2d, torch.float16),
    'float32': (fn_npu_fp32_2d, torch.float32),
    'bfloat16': (fn_npu_bf16_2d, torch.bfloat16),
    'bool': (fn_npu_bool_2d, torch.bool),
}
dtype_mapping1d = {
    'int8': (fn_npu_int8_1d, torch.int8),
    'int16': (fn_npu_int16_1d, torch.int16),
    'int32': (fn_npu_int32_1d, torch.int32),
    'int64': (fn_npu_int64_1d, torch.int64),
    'float16': (fn_npu_fp16_1d, torch.float16),
    'float32': (fn_npu_fp32_1d, torch.float32),
    'bfloat16': (fn_npu_bf16_1d, torch.bfloat16),
    'bool': (fn_npu_bool_1d, torch.bool),
}
dtype_mapping0d = {
    'int8': (fn_npu_int8_0d, torch.int8),
    'int16': (fn_npu_int16_0d, torch.int16),
    'int32': (fn_npu_int32_0d, torch.int32),
    'int64': (fn_npu_int64_0d, torch.int64),
    'float16': (fn_npu_fp16_0d, torch.float16),
    'float32': (fn_npu_fp32_0d, torch.float32),
    'bfloat16': (fn_npu_bf16_0d, torch.bfloat16),
    'bool': (fn_npu_bool_0d, torch.bool),
}

# 生成测试用例
testlist = [
    (func, sigtype, dtype, shape)
    for sigtype in test_dtype
    for shape in test_shape0d
    for func, dtype in [dtype_mapping0d[sigtype]]  # 直接解包映射结果
]

testlist += [
    (func, sigtype, dtype, shape)
    for sigtype in test_dtype
    for shape in test_shape1d
    for func, dtype in [dtype_mapping1d[sigtype]]  # 直接解包映射结果
]

testlist += [
    (func, sigtype, dtype, shape)
    for sigtype in test_dtype
    for shape in test_shape2d
    for func, dtype in [dtype_mapping2d[sigtype]]  # 直接解包映射结果
]

testlist += [
    (func, sigtype, dtype, shape)
    for sigtype in test_dtype
    for shape in test_shape3d
    for func, dtype in [dtype_mapping3d[sigtype]]  # 直接解包映射结果
]


@pytest.mark.parametrize('testfunc, sigtype, dtype, shape', testlist)
def test_npu(testfunc, sigtype, dtype, shape):
    if check_ub_mem_overflow(sigtype, shape):
        pytest.skip(f"dtype:{sigtype} shape:{shape} mem overflow")
    x = 0
    output = 0
    if len(shape) == 3:
        x = torch.full((shape[0], shape[1], shape[2]), 0, dtype=dtype).npu()
        output = torch.randint(1, (shape[0], shape[1], shape[2]), dtype=dtype).npu()
        testfunc[(1, 1, 1)](output, shape[0], shape[1], shape[2], shape[0], shape[1], shape[2])
    if len(shape) == 2:
        x = torch.full((shape[0], shape[1]), 0, dtype=dtype).npu()
        output = torch.randint(1, (shape[0], shape[1]), dtype=dtype).npu()
        shape0 = shape[0]
        shape1 = shape[1]
        if x.numel() * x.element_size() >= 8192:
            grid = (shape0, 1, 1)
            shape0 = 1
        else:
            grid = (1, 1, 1)
        testfunc[grid](output, shape0, shape1, shape0, shape1)
    if len(shape) == 1:
        x = torch.full((shape[0],), 0, dtype=dtype).npu()
        output = torch.randint(1, (shape[0],), dtype=dtype).npu()
        testfunc[1, 1, 1](output, shape[0], shape[0])
    if len(shape) == 0:
        output = torch.randint(1, size=shape, dtype=dtype).npu()
        x = torch.zeros_like(output)
        testfunc[(1,)](output_ptr=output, N=1)
    test_common.validate_cmp(sigtype, output, x)


@triton.jit
def fn_npu_multi_d(output_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr,
                   NB: tl.constexpr):
    dtype = output_ptr.type.element_ty

    offsets = tl.arange(0, XB) * (YB * ZB * MB * NB)
    if (YB * ZB * MB * NB) > 1:
        offsets = offsets[:, None] + tl.arange(0, YB)[None, :] * (ZB * MB * NB)
    if (ZB * MB * NB) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, ZB)[None, None, :] * (MB * NB)
    if (MB * NB) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, MB)[None, None, None, :] * NB
    if NB > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, NB)[None, None, None, None, :]

    if (YB * ZB * MB * NB) == 1:
        ret = tl.zeros((XB,), dtype=dtype)
    elif (ZB * MB * NB) == 1:
        ret = tl.zeros((XB, YB), dtype=dtype)
    elif (MB * NB) == 1:
        ret = tl.zeros((XB, YB, ZB), dtype=dtype)
    elif NB == 1:
        ret = tl.zeros((XB, YB, ZB, MB), dtype=dtype)
    else:
        ret = tl.zeros((XB, YB, ZB, MB, NB), dtype=dtype)

    tl.store(output_ptr + offsets, ret)


@pytest.mark.shape_4d_5d
@pytest.mark.parametrize('param_list',
                         [
                             ('float32', (4, 2, 16, 16)),
                             ('float32', (2, 4, 2, 16, 16)),

                             ('float32', (4, 2, 16, 16)),
                             ('float32', (2, 4, 2, 16, 16)),

                             ('float32', (4, 2, 16, 16)),
                             ('float32', (2, 4, 2, 16, 16)),
                         ]
                         )
def test_case_4d_5d(param_list):
    dtype, shape = param_list
    if check_ub_mem_overflow(dtype, shape):
        pytest.skip(f"dtype:{dtype} shape:{shape} mem overflow")
    y_ref = torch.full(shape, 0, dtype=eval('torch.' + dtype)).npu()
    print(f"y_ref = {torch.flatten(y_ref)[0:4]}")

    y_cal = torch.randint(1, shape, dtype=eval('torch.' + dtype)).npu()
    triton_shape = [*shape]
    while len(triton_shape) < 5:
        triton_shape.append(1)
    fn_npu_multi_d[(1,)](y_cal, *triton_shape)
    print(f"y_cal = {torch.flatten(y_cal)[0:4]}")
    test_common.validate_cmp(dtype, y_cal, y_ref)
