# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import triton
import triton.language as tl
import test_common

from test_common import TestUtils
import torch
import torch_npu
import pytest
import math
import random

@triton.jit
def fn_npu_int8_3d(output_ptr, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr):
    xidx = tl.arange(0, X)
    yidx = tl.arange(0, Y)
    zidx = tl.arange(0, Z)
    ret = tl.full((X, Y, Z), value=100, dtype=tl.int8)
    oidx = xidx[:, None, None] * Y * Z + yidx[None, :, None] * Z + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_int16_3d(output_ptr, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr):
    xidx = tl.arange(0, X)
    yidx = tl.arange(0, Y)
    zidx = tl.arange(0, Z)
    ret = tl.full((X, Y, Z), value=100, dtype=tl.int16)
    oidx = xidx[:, None, None] * Y * Z + yidx[None, :, None] * Z + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_uint32_3d(output_ptr, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr):
    xidx = tl.arange(0, X)
    yidx = tl.arange(0, Y)
    zidx = tl.arange(0, Z)
    ret = tl.full((X, Y, Z), value=100, dtype=tl.uint32)
    oidx = xidx[:, None, None] * Y * Z + yidx[None, :, None] * Z + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_int32_3d(output_ptr, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr):
    xidx = tl.arange(0, X)
    yidx = tl.arange(0, Y)
    zidx = tl.arange(0, Z)
    ret = tl.full((X, Y, Z), value=100, dtype=tl.int32)
    oidx = xidx[:, None, None] * Y * Z + yidx[None, :, None] * Z + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_int64_3d(output_ptr, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr):
    xidx = tl.arange(0, X)
    yidx = tl.arange(0, Y)
    zidx = tl.arange(0, Z)
    ret = tl.full((X, Y, Z), value=100, dtype=tl.int64)
    oidx = xidx[:, None, None] * Y * Z + yidx[None, :, None] * Z + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_fp16_3d(output_ptr, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr):
    xidx = tl.arange(0, X)
    yidx = tl.arange(0, Y)
    zidx = tl.arange(0, Z)
    ret = tl.full((X, Y, Z), value=100, dtype=tl.float16)
    oidx = xidx[:, None, None] * Y * Z + yidx[None, :, None] * Z + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_fp32_3d(output_ptr, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr):
    xidx = tl.arange(0, X)
    yidx = tl.arange(0, Y)
    zidx = tl.arange(0, Z)
    ret = tl.full((X, Y, Z), value=100, dtype=tl.float32)
    oidx = xidx[:, None, None] * Y * Z + yidx[None, :, None] * Z + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_bf16_3d(output_ptr, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr):
    xidx = tl.arange(0, X)
    yidx = tl.arange(0, Y)
    zidx = tl.arange(0, Z)
    ret = tl.full((X, Y, Z), value=100, dtype=tl.bfloat16)
    oidx = xidx[:, None, None] * Y * Z + yidx[None, :, None] * Z + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_bool_3d(output_ptr, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr):
    xidx = tl.arange(0, X)
    yidx = tl.arange(0, Y)
    zidx = tl.arange(0, Z)
    ret = tl.full((X, Y, Z), value=0, dtype=tl.int1)
    oidx = xidx[:, None, None] * Y * Z + yidx[None, :, None] * Z + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_int8_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr):
    yoffs = tl.program_id(0) * Y
    yidx = tl.arange(0, Y) + yoffs
    zidx = tl.arange(0, Z)
    ret = tl.full((Y, Z), value=100, dtype=tl.int8)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_int16_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr):
    yoffs = tl.program_id(0) * Y
    yidx = tl.arange(0, Y) + yoffs
    zidx = tl.arange(0, Z)
    ret = tl.full((Y, Z), value=100, dtype=tl.int16)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_uint32_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr):
    yoffs = tl.program_id(0) * Y
    yidx = tl.arange(0, Y) + yoffs
    zidx = tl.arange(0, Z)
    ret = tl.full((Y, Z), value=100, dtype=tl.uint32)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_int32_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr):
    yoffs = tl.program_id(0) * Y
    yidx = tl.arange(0, Y) + yoffs
    zidx = tl.arange(0, Z)
    ret = tl.full((Y, Z), value=100, dtype=tl.int32)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_int64_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr):
    yoffs = tl.program_id(0) * Y
    yidx = tl.arange(0, Y) + yoffs
    zidx = tl.arange(0, Z)
    ret = tl.full((Y, Z), value=100, dtype=tl.int64)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_fp16_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr):
    yoffs = tl.program_id(0) * Y
    yidx = tl.arange(0, Y) + yoffs
    zidx = tl.arange(0, Z)
    ret = tl.full((Y, Z), value=100, dtype=tl.float16)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_fp32_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr):
    yoffs = tl.program_id(0) * Y
    yidx = tl.arange(0, Y) + yoffs
    zidx = tl.arange(0, Z)
    ret = tl.full((Y, Z), value=100, dtype=tl.float32)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_bf16_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr):
    yoffs = tl.program_id(0) * Y
    yidx = tl.arange(0, Y) + yoffs
    zidx = tl.arange(0, Z)
    ret = tl.full((Y, Z), value=100, dtype=tl.bfloat16)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_bool_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr):
    yoffs = tl.program_id(0) * Y
    yidx = tl.arange(0, Y) + yoffs
    zidx = tl.arange(0, Z)
    ret = tl.full((Y, Z), value=0, dtype=tl.int1)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_int8_1d(output_ptr, Z: tl.constexpr):
    zidx = tl.arange(0, Z)
    ret = tl.full((Z,), value=100, dtype=tl.int8)
    oidx = zidx
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_int16_1d(output_ptr, Z: tl.constexpr):
    zidx = tl.arange(0, Z)
    ret = tl.full((Z,), value=100, dtype=tl.int16)
    oidx = zidx
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_uint32_1d(output_ptr, Z: tl.constexpr):
    zidx = tl.arange(0, Z)
    ret = tl.full((Z,), value=100, dtype=tl.uint32)
    oidx = zidx
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_int32_1d(output_ptr, Z: tl.constexpr):
    zidx = tl.arange(0, Z)
    ret = tl.full((Z,), value=100, dtype=tl.int32)
    oidx = zidx
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_int64_1d(output_ptr, Z: tl.constexpr):
    zidx = tl.arange(0, Z)
    ret = tl.full((Z,), value=100, dtype=tl.int64)
    oidx = zidx
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_fp16_1d(output_ptr, Z: tl.constexpr):
    zidx = tl.arange(0, Z)
    ret = tl.full((Z,), value=100, dtype=tl.float16)
    oidx = zidx
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_fp32_1d(output_ptr, Z: tl.constexpr):
    zidx = tl.arange(0, Z)
    ret = tl.full((Z,), value=100, dtype=tl.float32)
    oidx = zidx
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_bf16_1d(output_ptr, Z: tl.constexpr):
    zidx = tl.arange(0, Z)
    ret = tl.full((Z,), value=100, dtype=tl.bfloat16)
    oidx = zidx
    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_bool_1d(output_ptr, Z: tl.constexpr):
    zidx = tl.arange(0, Z)
    ret = tl.full((Z,), value=0, dtype=tl.int1)
    oidx = zidx
    tl.store(output_ptr + oidx, ret)


test_dtype = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'bool']
test_shape1d = TestUtils.test_shape1d
test_shape2d = TestUtils.test_shape2d
test_shape3d = TestUtils.test_shape3d

# 定义 dtype 到 (test_func, test_sigtype) 的映射
dtype_mapping3d = {
    'int8': (fn_npu_int8_3d, torch.int8),
    'int16': (fn_npu_int16_3d, torch.int16),
    'int32': (fn_npu_int32_3d, torch.int32),
    'uint32': (fn_npu_uint32_3d, torch.uint32),
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
    'uint32': (fn_npu_uint32_2d, torch.uint32),
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
    'uint32': (fn_npu_uint32_1d, torch.uint32),
    'int64': (fn_npu_int64_1d, torch.int64),
    'float16': (fn_npu_fp16_1d, torch.float16),
    'float32': (fn_npu_fp32_1d, torch.float32),
    'bfloat16': (fn_npu_bf16_1d, torch.bfloat16),
    'bool': (fn_npu_bool_1d, torch.bool),
}

# 生成测试用例
testlist = [
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
    x = 0
    output = 0
    if len(shape) == 3:
        if dtype == torch.bool:
            x = torch.full((shape[0], shape[1], shape[2]), 0, dtype=dtype).npu()
        else:
            x = torch.full((shape[0], shape[1], shape[2]), 100, dtype=dtype).npu()
        output = torch.randint(1, (shape[0], shape[1], shape[2]), dtype=dtype).npu()
        testfunc[(1, 1, 1)](output, shape[0], shape[1], shape[2], debug=True)
    if len(shape) == 2:
        if dtype == torch.bool:
            x = torch.full((shape[0], shape[1]), 0, dtype=dtype).npu()
        else:
            x = torch.full((shape[0], shape[1]), 100, dtype=dtype).npu()
        output = torch.randint(1, (shape[0], shape[1]), dtype=dtype).npu()
        shape0 = shape[0]
        shape1 = shape[1]
        if x.numel() * x.element_size() >= 8192:
            grid = (shape0, 1, 1)
            shape0 = 1
        else:
            grid = (1, 1, 1)
        testfunc[grid](output, shape0, shape1, debug=True)
    if len(shape) == 1:
        if dtype == torch.bool:
            x = torch.full((shape[0], ), 0, dtype=dtype).npu()
        else:
            x = torch.full((shape[0], ), 100, dtype=dtype).npu()
        output = torch.randint(1, (shape[0],), dtype=dtype).npu()
        testfunc[1, 1, 1](output, shape[0], debug=True)
    test_common.validate_cmp(sigtype, output, x)


@triton.jit
def fn_npu_multi_d(output_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr, NB: tl.constexpr):
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
        ret = tl.full((XB, ), value=100, dtype=dtype)
    elif (ZB * MB * NB) == 1:
        ret = tl.full((XB, YB), value=100, dtype=dtype)
    elif (MB * NB) == 1:
        ret = tl.full((XB, YB, ZB), value=100, dtype=dtype)
    elif NB == 1:
        ret = tl.full((XB, YB, ZB, MB), value=100, dtype=dtype)
    else:
        ret = tl.full((XB, YB, ZB, MB, NB), value=100, dtype=dtype)

    tl.store(output_ptr + offsets, ret)


testlist_multi_d = [
    (fn_npu_multi_d, 'float32', torch.float32, (4, 2, 16, 16)),
    (fn_npu_multi_d, 'float32', torch.float32, (2, 4, 2, 16, 16)),

    (fn_npu_multi_d, 'float32', torch.float16, (4, 2, 16, 16)),
    (fn_npu_multi_d, 'float32', torch.float16, (2, 4, 2, 16, 16)),

    (fn_npu_multi_d, 'float32', torch.int8, (4, 2, 16, 16)),
    (fn_npu_multi_d, 'float32', torch.int8, (2, 4, 2, 16, 16)),
]


@pytest.mark.shape_4d_5d
@pytest.mark.parametrize('testfunc, sigtype, dtype, shape', testlist_multi_d)
def test_npu_4d_5d(testfunc, sigtype, dtype, shape):
    x = torch.full(shape, 100, dtype=dtype).npu()

    print(f"shape = {x.shape}")
    print(x.dtype)
    print(torch.flatten(x)[0:16])

    output = torch.randint(1, shape, dtype=dtype).npu()

    print(f"output.dtype={output.dtype}")

    triton_shape = [*shape]
    while len(triton_shape) < 5:
        triton_shape.append(1)
    testfunc[(1,)](output, *triton_shape)
    print(torch.flatten(output)[0:16])

    test_common.validate_cmp(sigtype, output, x)


@triton.jit
def fn_npu_bf16_6d(output_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                   D: tl.constexpr, E: tl.constexpr, F: tl.constexpr):
    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)
    fidx = tl.arange(0, F)

    ret = tl.full((A, B, C, D, E, F), value=10, dtype=tl.bfloat16)

    oidx = (aidx[:, None, None, None, None, None] *
            B * C * D * E * F +
            bidx[None, :, None, None, None, None] *
            C * D * E * F +
            cidx[None, None, :, None, None, None] *
            D * E * F +
            didx[None, None, None, :, None, None] *
            E * F +
            eidx[None, None, None, None, :, None] *
            F +
            fidx[None, None, None, None, None, :])

    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_int8_6d(output_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                   D: tl.constexpr, E: tl.constexpr, F: tl.constexpr):
    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)
    fidx = tl.arange(0, F)

    ret = tl.full((A, B, C, D, E, F), value=10, dtype=tl.int8)

    oidx = (aidx[:, None, None, None, None, None] *
            B * C * D * E * F +
            bidx[None, :, None, None, None, None] *
            C * D * E * F +
            cidx[None, None, :, None, None, None] *
            D * E * F +
            didx[None, None, None, :, None, None] *
            E * F +
            eidx[None, None, None, None, :, None] *
            F +
            fidx[None, None, None, None, None, :])

    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_int16_6d(output_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                    D: tl.constexpr, E: tl.constexpr, F: tl.constexpr):
    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)
    fidx = tl.arange(0, F)

    ret = tl.full((A, B, C, D, E, F), value=10, dtype=tl.int16)

    oidx = (aidx[:, None, None, None, None, None] *
            B * C * D * E * F +
            bidx[None, :, None, None, None, None] *
            C * D * E * F +
            cidx[None, None, :, None, None, None] *
            D * E * F +
            didx[None, None, None, :, None, None] *
            E * F +
            eidx[None, None, None, None, :, None] *
            F +
            fidx[None, None, None, None, None, :])

    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_int32_6d(output_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                    D: tl.constexpr, E: tl.constexpr, F: tl.constexpr):
    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)
    fidx = tl.arange(0, F)

    ret = tl.full((A, B, C, D, E, F), value=10, dtype=tl.int32)

    oidx = (aidx[:, None, None, None, None, None] *
            B * C * D * E * F +
            bidx[None, :, None, None, None, None] *
            C * D * E * F +
            cidx[None, None, :, None, None, None] *
            D * E * F +
            didx[None, None, None, :, None, None] *
            E * F +
            eidx[None, None, None, None, :, None] *
            F +
            fidx[None, None, None, None, None, :])

    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_int64_6d(output_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                    D: tl.constexpr, E: tl.constexpr, F: tl.constexpr):
    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)
    fidx = tl.arange(0, F)

    ret = tl.full((A, B, C, D, E, F), value=10, dtype=tl.int64)

    oidx = (aidx[:, None, None, None, None, None] *
            B * C * D * E * F +
            bidx[None, :, None, None, None, None] *
            C * D * E * F +
            cidx[None, None, :, None, None, None] *
            D * E * F +
            didx[None, None, None, :, None, None] *
            E * F +
            eidx[None, None, None, None, :, None] *
            F +
            fidx[None, None, None, None, None, :])

    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_fp16_6d(output_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                   D: tl.constexpr, E: tl.constexpr, F: tl.constexpr):
    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)
    fidx = tl.arange(0, F)

    ret = tl.full((A, B, C, D, E, F), value=10, dtype=tl.float16)

    oidx = (aidx[:, None, None, None, None, None] *
            B * C * D * E * F +
            bidx[None, :, None, None, None, None] *
            C * D * E * F +
            cidx[None, None, :, None, None, None] *
            D * E * F +
            didx[None, None, None, :, None, None] *
            E * F +
            eidx[None, None, None, None, :, None] *
            F +
            fidx[None, None, None, None, None, :])

    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_fp32_6d(output_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                   D: tl.constexpr, E: tl.constexpr, F: tl.constexpr):
    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)
    fidx = tl.arange(0, F)

    ret = tl.full((A, B, C, D, E, F), value=10, dtype=tl.float32)

    oidx = (aidx[:, None, None, None, None, None] *
            B * C * D * E * F +
            bidx[None, :, None, None, None, None] *
            C * D * E * F +
            cidx[None, None, :, None, None, None] *
            D * E * F +
            didx[None, None, None, :, None, None] *
            E * F +
            eidx[None, None, None, None, :, None] *
            F +
            fidx[None, None, None, None, None, :])

    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_bool_6d(output_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                   D: tl.constexpr, E: tl.constexpr, F: tl.constexpr):
    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)
    fidx = tl.arange(0, F)

    ret = tl.full((A, B, C, D, E, F), value=0, dtype=tl.int1)

    oidx = (aidx[:, None, None, None, None, None] *
            B * C * D * E * F +
            bidx[None, :, None, None, None, None] *
            C * D * E * F +
            cidx[None, None, :, None, None, None] *
            D * E * F +
            didx[None, None, None, :, None, None] *
            E * F +
            eidx[None, None, None, None, :, None] *
            F +
            fidx[None, None, None, None, None, :])

    tl.store(output_ptr + oidx, ret)


test_dtype = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'bool']
test_shape6d = TestUtils.test_shape6d
dtype_mapping6d = {
    'int8': (fn_npu_int8_6d, torch.int8),
    'int16': (fn_npu_int16_6d, torch.int16),
    'int32': (fn_npu_int32_6d, torch.int32),
    'int64': (fn_npu_int64_6d, torch.int64),
    'float16': (fn_npu_fp16_6d, torch.float16),
    'float32': (fn_npu_fp32_6d, torch.float32),
    'bfloat16': (fn_npu_bf16_6d, torch.bfloat16),
    'bool': (fn_npu_bool_6d, torch.bool),
}

testlist6d = [
    (func, sigtype, dtype, shape)
    for sigtype in test_dtype
    for shape in test_shape6d
    for func, dtype in [dtype_mapping6d[sigtype]]
]


@pytest.mark.parametrize('testfunc, sigtype, dtype, shape', testlist6d)
def test_npu_6d(testfunc, sigtype, dtype, shape):
    x = 0
    output = 0
    if len(shape) == 6:
        if dtype == torch.bool:
            x = torch.full((shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]), 0, dtype=dtype).npu()
        else:
            x = torch.full(shape, 10, dtype=dtype).npu()
        output = torch.randint(1, shape, dtype=dtype).npu()
        testfunc[1, 1, 1](output, *shape, debug=True)
    test_common.validate_cmp(sigtype, output, x)


@triton.jit
def fn_npu_bf16_7d(output_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                   D: tl.constexpr, E: tl.constexpr, F: tl.constexpr,
                   G: tl.constexpr):
    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)
    fidx = tl.arange(0, F)
    gidx = tl.arange(0, G)

    ret = tl.full((A, B, C, D, E, F, G), value=10, dtype=tl.bfloat16)

    oidx = (aidx[:, None, None, None, None, None, None] *
            B * C * D * E * F * G +
            bidx[None, :, None, None, None, None, None] *
            C * D * E * F * G +
            cidx[None, None, :, None, None, None, None] *
            D * E * F * G +
            didx[None, None, None, :, None, None, None] *
            E * F * G +
            eidx[None, None, None, None, :, None, None] *
            F * G +
            fidx[None, None, None, None, None, :, None] *
            G +
            gidx[None, None, None, None, None, None, :])

    tl.store(output_ptr + oidx, ret)

@triton.jit
def fn_npu_int8_7d(output_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                   D: tl.constexpr, E: tl.constexpr, F: tl.constexpr,
                   G: tl.constexpr):

    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)
    fidx = tl.arange(0, F)
    gidx = tl.arange(0, G)

    ret = tl.full((A, B, C, D, E, F, G), value=10, dtype=tl.int8)

    oidx = (aidx[:, None, None, None, None, None, None] *
            B * C * D * E * F * G +
            bidx[None, :, None, None, None, None, None] *
            C * D * E * F * G +
            cidx[None, None, :, None, None, None, None] *
            D * E * F * G +
            didx[None, None, None, :, None, None, None] *
            E * F * G +
            eidx[None, None, None, None, :, None, None] *
            F * G +
            fidx[None, None, None, None, None, :, None] *
            G +
            gidx[None, None, None, None, None, None, :])

    tl.store(output_ptr + oidx, ret)

@triton.jit
def fn_npu_int16_7d(output_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                   D: tl.constexpr, E: tl.constexpr, F: tl.constexpr,
                   G: tl.constexpr):

    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)
    fidx = tl.arange(0, F)
    gidx = tl.arange(0, G)

    ret = tl.full((A, B, C, D, E, F, G), value=10, dtype=tl.int16)

    oidx = (aidx[:, None, None, None, None, None, None] *
            B * C * D * E * F * G +
            bidx[None, :, None, None, None, None, None] *
            C * D * E * F * G +
            cidx[None, None, :, None, None, None, None] *
            D * E * F * G +
            didx[None, None, None, :, None, None, None] *
            E * F * G +
            eidx[None, None, None, None, :, None, None] *
            F * G +
            fidx[None, None, None, None, None, :, None] *
            G +
            gidx[None, None, None, None, None, None, :])

    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_int32_7d(output_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                   D: tl.constexpr, E: tl.constexpr, F: tl.constexpr,
                   G: tl.constexpr):

    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)
    fidx = tl.arange(0, F)
    gidx = tl.arange(0, G)

    ret = tl.full((A, B, C, D, E, F, G), value=10, dtype=tl.int32)

    oidx = (aidx[:, None, None, None, None, None, None] *
            B * C * D * E * F * G +
            bidx[None, :, None, None, None, None, None] *
            C * D * E * F * G +
            cidx[None, None, :, None, None, None, None] *
            D * E * F * G +
            didx[None, None, None, :, None, None, None] *
            E * F * G +
            eidx[None, None, None, None, :, None, None] *
            F * G +
            fidx[None, None, None, None, None, :, None] *
            G +
            gidx[None, None, None, None, None, None, :])

    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_int64_7d(output_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                   D: tl.constexpr, E: tl.constexpr, F: tl.constexpr,
                   G: tl.constexpr):

    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)
    fidx = tl.arange(0, F)
    gidx = tl.arange(0, G)

    ret = tl.full((A, B, C, D, E, F, G), value=10, dtype=tl.int64)

    oidx = (aidx[:, None, None, None, None, None, None] *
            B * C * D * E * F * G +
            bidx[None, :, None, None, None, None, None] *
            C * D * E * F * G +
            cidx[None, None, :, None, None, None, None] *
            D * E * F * G +
            didx[None, None, None, :, None, None, None] *
            E * F * G +
            eidx[None, None, None, None, :, None, None] *
            F * G +
            fidx[None, None, None, None, None, :, None] *
            G +
            gidx[None, None, None, None, None, None, :])

    tl.store(output_ptr + oidx, ret)

@triton.jit
def fn_npu_fp16_7d(output_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                   D: tl.constexpr, E: tl.constexpr, F: tl.constexpr,
                   G: tl.constexpr):

    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)
    fidx = tl.arange(0, F)
    gidx = tl.arange(0, G)

    ret = tl.full((A, B, C, D, E, F, G), value=10, dtype=tl.float16)

    oidx = (aidx[:, None, None, None, None, None, None] *
            B * C * D * E * F * G +
            bidx[None, :, None, None, None, None, None] *
            C * D * E * F * G +
            cidx[None, None, :, None, None, None, None] *
            D * E * F * G +
            didx[None, None, None, :, None, None, None] *
            E * F * G +
            eidx[None, None, None, None, :, None, None] *
            F * G +
            fidx[None, None, None, None, None, :, None] *
            G +
            gidx[None, None, None, None, None, None, :])

    tl.store(output_ptr + oidx, ret)

@triton.jit
def fn_npu_fp32_7d(output_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                   D: tl.constexpr, E: tl.constexpr, F: tl.constexpr,
                   G: tl.constexpr):

    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)
    fidx = tl.arange(0, F)
    gidx = tl.arange(0, G)

    ret = tl.full((A, B, C, D, E, F, G), value=10, dtype=tl.float32)

    oidx = (aidx[:, None, None, None, None, None, None] *
            B * C * D * E * F * G +
            bidx[None, :, None, None, None, None, None] *
            C * D * E * F * G +
            cidx[None, None, :, None, None, None, None] *
            D * E * F * G +
            didx[None, None, None, :, None, None, None] *
            E * F * G +
            eidx[None, None, None, None, :, None, None] *
            F * G +
            fidx[None, None, None, None, None, :, None] *
            G +
            gidx[None, None, None, None, None, None, :])

    tl.store(output_ptr + oidx, ret)

@triton.jit
def fn_npu_bool_7d(output_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                   D: tl.constexpr, E: tl.constexpr, F: tl.constexpr,
                   G: tl.constexpr):

    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)
    fidx = tl.arange(0, F)
    gidx = tl.arange(0, G)

    ret = tl.full((A, B, C, D, E, F, G), value=0, dtype=tl.int1)

    oidx = (aidx[:, None, None, None, None, None, None] *
            B * C * D * E * F * G +
            bidx[None, :, None, None, None, None, None] *
            C * D * E * F * G +
            cidx[None, None, :, None, None, None, None] *
            D * E * F * G +
            didx[None, None, None, :, None, None, None] *
            E * F * G +
            eidx[None, None, None, None, :, None, None] *
            F * G +
            fidx[None, None, None, None, None, :, None] *
            G +
            gidx[None, None, None, None, None, None, :])

    tl.store(output_ptr + oidx, ret)

test_dtype = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'bool']
test_shape7d = TestUtils.test_shape7d
dtype_mapping7d = {
    'int8': (fn_npu_int8_7d, torch.int8),
    'int16': (fn_npu_int16_7d, torch.int16),
    'int32': (fn_npu_int32_7d, torch.int32),
    'int64': (fn_npu_int64_7d, torch.int64),
    'float16': (fn_npu_fp16_7d, torch.float16),
    'float32': (fn_npu_fp32_7d, torch.float32),
    'bfloat16': (fn_npu_bf16_7d, torch.bfloat16),
    'bool': (fn_npu_bool_7d, torch.bool),
}

testlist7d = [
    (func, sigtype, dtype, shape)
    for sigtype in test_dtype
    for shape in test_shape7d
    for func, dtype in [dtype_mapping7d[sigtype]]
]

@pytest.mark.parametrize('testfunc, sigtype, dtype, shape', testlist7d)
def test_npu_7d(testfunc, sigtype, dtype, shape):
    x = 0
    output = 0
    if len(shape) == 7:
        if dtype == torch.bool:
            x = torch.full((shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6]), 0, dtype=dtype).npu()
        else:
            x = torch.full(shape, 10, dtype=dtype).npu()
        output = torch.randint(1, shape, dtype=dtype).npu()
        testfunc[1, 1, 1](output, *shape, debug=True)
    test_common.validate_cmp(sigtype, output, x)


@triton.jit
def fn_npu_bf16_8d(output_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                   D: tl.constexpr, E: tl.constexpr, F: tl.constexpr,
                   G: tl.constexpr, H: tl.constexpr):
    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)
    fidx = tl.arange(0, F)
    gidx = tl.arange(0, G)
    hidx = tl.arange(0, H)

    ret = tl.full((A, B, C, D, E, F, G, H), value=10, dtype=tl.bfloat16)

    oidx = (aidx[:, None, None, None, None, None, None, None] *
            B * C * D * E * F * G * H +
            bidx[None, :, None, None, None, None, None, None] *
            C * D * E * F * G * H +
            cidx[None, None, :, None, None, None, None, None] *
            D * E * F * G * H +
            didx[None, None, None, :, None, None, None, None] *
            E * F * G * H +
            eidx[None, None, None, None, :, None, None, None] *
            F * G * H +
            fidx[None, None, None, None, None, :, None, None] *
            G * H +
            gidx[None, None, None, None, None, None, :, None] *
            H +
            hidx[None, None, None, None, None, None, None, :])

    tl.store(output_ptr + oidx, ret)

@triton.jit
def fn_npu_int8_8d(output_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                   D: tl.constexpr, E: tl.constexpr, F: tl.constexpr,
                   G: tl.constexpr, H: tl.constexpr):

    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)
    fidx = tl.arange(0, F)
    gidx = tl.arange(0, G)
    hidx = tl.arange(0, H)

    ret = tl.full((A, B, C, D, E, F, G, H), value=10, dtype=tl.int8)

    oidx = (aidx[:, None, None, None, None, None, None, None] *
            B * C * D * E * F * G * H +
            bidx[None, :, None, None, None, None, None, None] *
            C * D * E * F * G * H +
            cidx[None, None, :, None, None, None, None, None] *
            D * E * F * G * H +
            didx[None, None, None, :, None, None, None, None] *
            E * F * G * H +
            eidx[None, None, None, None, :, None, None, None] *
            F * G * H +
            fidx[None, None, None, None, None, :, None, None] *
            G * H +
            gidx[None, None, None, None, None, None, :, None] *
            H +
            hidx[None, None, None, None, None, None, None, :])

    tl.store(output_ptr + oidx, ret)

@triton.jit
def fn_npu_int16_8d(output_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                   D: tl.constexpr, E: tl.constexpr, F: tl.constexpr,
                   G: tl.constexpr, H: tl.constexpr):

    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)
    fidx = tl.arange(0, F)
    gidx = tl.arange(0, G)
    hidx = tl.arange(0, H)

    ret = tl.full((A, B, C, D, E, F, G, H), value=10, dtype=tl.int16)

    oidx = (aidx[:, None, None, None, None, None, None, None] *
            B * C * D * E * F * G * H +
            bidx[None, :, None, None, None, None, None, None] *
            C * D * E * F * G * H +
            cidx[None, None, :, None, None, None, None, None] *
            D * E * F * G * H +
            didx[None, None, None, :, None, None, None, None] *
            E * F * G * H +
            eidx[None, None, None, None, :, None, None, None] *
            F * G * H +
            fidx[None, None, None, None, None, :, None, None] *
            G * H +
            gidx[None, None, None, None, None, None, :, None] *
            H +
            hidx[None, None, None, None, None, None, None, :])

    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_int32_8d(output_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                   D: tl.constexpr, E: tl.constexpr, F: tl.constexpr,
                   G: tl.constexpr, H: tl.constexpr):

    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)
    fidx = tl.arange(0, F)
    gidx = tl.arange(0, G)
    hidx = tl.arange(0, H)

    ret = tl.full((A, B, C, D, E, F, G, H), value=10, dtype=tl.int32)

    oidx = (aidx[:, None, None, None, None, None, None, None] *
            B * C * D * E * F * G * H +
            bidx[None, :, None, None, None, None, None, None] *
            C * D * E * F * G * H +
            cidx[None, None, :, None, None, None, None, None] *
            D * E * F * G * H +
            didx[None, None, None, :, None, None, None, None] *
            E * F * G * H +
            eidx[None, None, None, None, :, None, None, None] *
            F * G * H +
            fidx[None, None, None, None, None, :, None, None] *
            G * H +
            gidx[None, None, None, None, None, None, :, None] *
            H +
            hidx[None, None, None, None, None, None, None, :])

    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_int64_8d(output_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                   D: tl.constexpr, E: tl.constexpr, F: tl.constexpr,
                   G: tl.constexpr, H: tl.constexpr):

    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)
    fidx = tl.arange(0, F)
    gidx = tl.arange(0, G)
    hidx = tl.arange(0, H)

    ret = tl.full((A, B, C, D, E, F, G, H), value=10, dtype=tl.int64)

    oidx = (aidx[:, None, None, None, None, None, None, None] *
            B * C * D * E * F * G * H +
            bidx[None, :, None, None, None, None, None, None] *
            C * D * E * F * G * H +
            cidx[None, None, :, None, None, None, None, None] *
            D * E * F * G * H +
            didx[None, None, None, :, None, None, None, None] *
            E * F * G * H +
            eidx[None, None, None, None, :, None, None, None] *
            F * G * H +
            fidx[None, None, None, None, None, :, None, None] *
            G * H +
            gidx[None, None, None, None, None, None, :, None] *
            H +
            hidx[None, None, None, None, None, None, None, :])

    tl.store(output_ptr + oidx, ret)

@triton.jit
def fn_npu_fp16_8d(output_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                   D: tl.constexpr, E: tl.constexpr, F: tl.constexpr,
                   G: tl.constexpr, H: tl.constexpr):

    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)
    fidx = tl.arange(0, F)
    gidx = tl.arange(0, G)
    hidx = tl.arange(0, H)

    ret = tl.full((A, B, C, D, E, F, G, H), value=10, dtype=tl.float16)

    oidx = (aidx[:, None, None, None, None, None, None, None] *
            B * C * D * E * F * G * H +
            bidx[None, :, None, None, None, None, None, None] *
            C * D * E * F * G * H +
            cidx[None, None, :, None, None, None, None, None] *
            D * E * F * G * H +
            didx[None, None, None, :, None, None, None, None] *
            E * F * G * H +
            eidx[None, None, None, None, :, None, None, None] *
            F * G * H +
            fidx[None, None, None, None, None, :, None, None] *
            G * H +
            gidx[None, None, None, None, None, None, :, None] *
            H +
            hidx[None, None, None, None, None, None, None, :])

    tl.store(output_ptr + oidx, ret)

@triton.jit
def fn_npu_fp32_8d(output_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                   D: tl.constexpr, E: tl.constexpr, F: tl.constexpr,
                   G: tl.constexpr, H: tl.constexpr):

    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)
    fidx = tl.arange(0, F)
    gidx = tl.arange(0, G)
    hidx = tl.arange(0, H)

    ret = tl.full((A, B, C, D, E, F, G, H), value=10, dtype=tl.float32)

    oidx = (aidx[:, None, None, None, None, None, None, None] *
            B * C * D * E * F * G * H +
            bidx[None, :, None, None, None, None, None, None] *
            C * D * E * F * G * H +
            cidx[None, None, :, None, None, None, None, None] *
            D * E * F * G * H +
            didx[None, None, None, :, None, None, None, None] *
            E * F * G * H +
            eidx[None, None, None, None, :, None, None, None] *
            F * G * H +
            fidx[None, None, None, None, None, :, None, None] *
            G * H +
            gidx[None, None, None, None, None, None, :, None] *
            H +
            hidx[None, None, None, None, None, None, None, :])

    tl.store(output_ptr + oidx, ret)

@triton.jit
def fn_npu_bool_8d(output_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                   D: tl.constexpr, E: tl.constexpr, F: tl.constexpr,
                   G: tl.constexpr, H: tl.constexpr):

    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)
    fidx = tl.arange(0, F)
    gidx = tl.arange(0, G)
    hidx = tl.arange(0, H)

    ret = tl.full((A, B, C, D, E, F, G, H), value=0, dtype=tl.int1)

    oidx = (aidx[:, None, None, None, None, None, None, None] *
            B * C * D * E * F * G * H +
            bidx[None, :, None, None, None, None, None, None] *
            C * D * E * F * G * H +
            cidx[None, None, :, None, None, None, None, None] *
            D * E * F * G * H +
            didx[None, None, None, :, None, None, None, None] *
            E * F * G * H +
            eidx[None, None, None, None, :, None, None, None] *
            F * G * H +
            fidx[None, None, None, None, None, :, None, None] *
            G * H +
            gidx[None, None, None, None, None, None, :, None] *
            H +
            hidx[None, None, None, None, None, None, None, :])

    tl.store(output_ptr + oidx, ret)

test_dtype = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'bool']
test_shape8d = TestUtils.test_shape8d
dtype_mapping8d = {
    'int8': (fn_npu_int8_8d, torch.int8),
    'int16': (fn_npu_int16_8d, torch.int16),
    'int32': (fn_npu_int32_8d, torch.int32),
    'int64': (fn_npu_int64_8d, torch.int64),
    'float16': (fn_npu_fp16_8d, torch.float16),
    'float32': (fn_npu_fp32_8d, torch.float32),
    'bfloat16': (fn_npu_bf16_8d, torch.bfloat16),
    'bool': (fn_npu_bool_8d, torch.bool),
}

testlist8d = [
    (func, sigtype, dtype, shape)
    for sigtype in test_dtype
    for shape in test_shape8d
    for func, dtype in [dtype_mapping8d[sigtype]]
]


@pytest.mark.parametrize('testfunc, sigtype, dtype, shape', testlist8d)
def test_npu_8d(testfunc, sigtype, dtype, shape):
    x = 0
    output = 0
    if len(shape) == 8:
        if dtype == torch.bool:
            x = torch.full((shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7]), 0, dtype=dtype).npu()
        else:
            x = torch.full(shape, 10, dtype=dtype).npu()
        output = torch.randint(1, shape, dtype=dtype).npu()
        testfunc[1, 1, 1](output, *shape, debug=True)
    test_common.validate_cmp(sigtype, output, x)
