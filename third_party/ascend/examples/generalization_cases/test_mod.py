import logging

import triton
import triton.language as tl
import torch
import pytest
import test_common
from test_common import TestUtils
import math


def torch_pointwise(x, y):
    res = x % y
    return res


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
    Y = tl.load(y_ptr + idx)

    ret = X % Y

    tl.store(output_ptr + idx, ret)


@triton.jit
def triton_mod_4d(
        output_ptr, x_ptr, y_ptr,
        BLOCK_SIZE: tl.constexpr, SUB_BLOCK: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr, SHAPE_3: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr, STRIDE_3: tl.constexpr,
):
    pid = tl.program_id(0)
    for loop in range(0, tl.cdiv(BLOCK_SIZE, SUB_BLOCK)):
        base_idx = tl.arange(0, SUB_BLOCK)
        pid_tensor = tl.full((SUB_BLOCK,), pid * BLOCK_SIZE + loop * SUB_BLOCK, dtype=tl.int32)
        tmp0 = (pid_tensor + base_idx)[:, None, None, None]
        tmp1 = tl.arange(0, SHAPE_1)[None, :, None, None]
        tmp2 = tl.arange(0, SHAPE_2)[None, None, :, None]
        tmp3 = tl.arange(0, SHAPE_3)[None, None, None, :]
        offsets = tmp0 * STRIDE_0 + tmp1 * STRIDE_1 + tmp2 * STRIDE_2 + tmp3 * STRIDE_3
        masks = tmp0 < SHAPE_0
        x = tl.load(x_ptr + offsets, mask=masks)
        y = tl.load(y_ptr + offsets, mask=masks)
        ret = x % y
        tl.store(output_ptr + offsets, ret, mask=masks)


@triton.jit
def triton_mod_5d(
        output_ptr, x_ptr, y_ptr,
        BLOCK_SIZE: tl.constexpr, SUB_BLOCK: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr, SHAPE_3: tl.constexpr,
        SHAPE_4: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr, STRIDE_3: tl.constexpr,
        STRIDE_4: tl.constexpr,
):
    pid = tl.program_id(0)
    for loop in range(0, tl.cdiv(BLOCK_SIZE, SUB_BLOCK)):
        base_idx = tl.arange(0, SUB_BLOCK)
        pid_tensor = tl.full((SUB_BLOCK,), pid * BLOCK_SIZE + loop * SUB_BLOCK, dtype=tl.int32)
        tmp0 = (pid_tensor + base_idx)[:, None, None, None, None]
        tmp1 = tl.arange(0, SHAPE_1)[None, :, None, None, None]
        tmp2 = tl.arange(0, SHAPE_2)[None, None, :, None, None]
        tmp3 = tl.arange(0, SHAPE_3)[None, None, None, :, None]
        tmp4 = tl.arange(0, SHAPE_4)[None, None, None, None, :]
        offsets = tmp0 * STRIDE_0 + tmp1 * STRIDE_1 + tmp2 * STRIDE_2 + tmp3 * STRIDE_3 + tmp4 * STRIDE_4
        masks = tmp0 < SHAPE_0
        x = tl.load(x_ptr + offsets, mask=masks)
        y = tl.load(y_ptr + offsets, mask=masks)
        ret = x % y
        tl.store(output_ptr + offsets, ret, mask=masks)


@pytest.mark.parametrize('shape', TestUtils.test_shape1_2_3d)
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'bfloat16', 'int8', 'int16', 'int32', 'int64'])
def test_case2(dtype, shape):
    if dtype in ['int8', 'int16', 'int32', 'int64']:
        x = test_common.generate_tensor_int_withSigns(shape, dtype).npu()
        y = test_common.generate_tensor_int_withSigns(shape, dtype).npu()
        z = test_common.generate_tensor_int_withSigns(shape, dtype).npu()
    else:
        x = test_common.generate_tensor(shape, dtype).npu()
        y = test_common.generate_tensor(shape, dtype).npu()
        z = test_common.generate_tensor(shape, dtype).npu()

    x[x <= 0] = 1
    y[y <= 0] = 1
    z[z <= 0] = 1

    ans = torch_pointwise(x.cpu(), y.cpu())
    ans = ans.npu()
    output = torch.zeros_like(ans)

    if len(shape) == 1:
        fn_npu_[1, 1, shape[0]](output, x, y, z, 1, 1, 1, 1, 1, shape[0])
    elif len(shape) == 2:
        if shape[0] > shape[1]:
            fn_npu_[1, shape[0], 1](output, x, y, z, 1, 1, shape[1], 1, shape[0], shape[1])
        else:
            fn_npu_[1, 1, shape[1]](output, x, y, z, 1, shape[0], 1, 1, shape[0], shape[1])
    elif len(shape) == 3:
        if max(shape[0], shape[1], shape[2]) == shape[0]:
            fn_npu_[shape[0], 1, 1](output, x, y, z, 1, shape[1], shape[2], shape[0], shape[1], shape[2])
        elif max(shape[0], shape[1], shape[2]) == shape[1]:
            fn_npu_[1, shape[1], 1](output, x, y, z, shape[0], 1, shape[2], shape[0], shape[1], shape[2])
        else:
            fn_npu_[1, 1, shape[2]](output, x, y, z, shape[0], shape[1], 1, shape[0], shape[1], shape[2])
    else:
        fn_npu_[1, 1, 1](output, x, y, z, 1, 1, 1, 1, 1, 1)

    test_common.validate_cmp(dtype, ans, output)


invalid_types = [
    'bool',
]


@pytest.mark.parametrize("sigtype", invalid_types)
@test_common.raises_with_match(triton.compiler.errors.CompilationError, "unexpected type")
def test_invalid_types(sigtype):
    N = 32
    x = test_common.generate_tensor(shape=(N,), dtype=sigtype).npu()
    y = test_common.generate_tensor(shape=(N,), dtype=sigtype).npu()
    z = test_common.generate_tensor(shape=(N,), dtype=sigtype).npu()
    output = test_common.generate_tensor(shape=(N,), dtype=sigtype).npu()

    fn_npu_[1, 1, 1](output, x, y, z, 32, 1, 1, 32, 1, 1)


@pytest.mark.parametrize('shape',
                         TestUtils.test_shape4d + [(25, 2, 3, 31), (2, 2, 39, 23), (17, 27, 3, 3), (3, 2, 27, 37)])
@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16'])
def test_mod_4d(shape, dtype):
    logging.log(logging.DEBUG, f"shape = {shape}")
    if dtype in ['int8', 'int16', 'int32', 'int64']:
        x = test_common.generate_tensor_int_withSigns(shape, dtype).npu()
        y = test_common.generate_tensor_int_withSigns(shape, dtype).npu()
    else:
        x = test_common.generate_tensor(shape, dtype).npu()
        y = test_common.generate_tensor(shape, dtype).npu()

    x[x <= 0] = 1
    y[y <= 0] = 1

    output = torch.randint(1, shape, dtype=eval('torch.' + dtype)).npu()
    logging.log(logging.DEBUG, f"output.dtype={output.dtype}")

    ans = torch_pointwise(x.cpu(), y.cpu())
    ans = ans.npu()

    n = x.numel()
    block_size = min(triton.next_power_of_2(n), 64)
    sub_block_size = 1
    grid = (triton.cdiv(n, block_size),)
    print(" ")
    print(f"=== loops: {triton.cdiv(block_size, sub_block_size)}")
    print(f"=== grid : {grid}")
    triton_mod_4d[grid](output, x, y, block_size, sub_block_size, *list(shape), *list(x.stride()))

    test_common.validate_cmp(dtype, ans, output)


@pytest.mark.parametrize('shape', TestUtils.test_shape5d + [(32, 5, 3, 1, 8)])
@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16'])
def test_mod_5d(shape, dtype):
    logging.log(logging.DEBUG, f"shape = {shape}")
    if dtype in ['int8', 'int16', 'int32', 'int64']:
        x = test_common.generate_tensor_int_withSigns(shape, dtype).npu()
        y = test_common.generate_tensor_int_withSigns(shape, dtype).npu()
    else:
        x = test_common.generate_tensor(shape, dtype).npu()
        y = test_common.generate_tensor(shape, dtype).npu()

    x[x <= 0] = 1
    y[y <= 0] = 1

    output = torch.randint(1, shape, dtype=eval('torch.' + dtype)).npu()
    logging.log(logging.DEBUG, f"output.dtype={output.dtype}")

    ans = torch_pointwise(x.cpu(), y.cpu())
    ans = ans.npu()

    n = x.numel()
    block_size = min(triton.next_power_of_2(n), 32)
    sub_block_size = 1
    grid = (triton.cdiv(n, block_size),)
    print(" ")
    print(f"=== loops: {triton.cdiv(block_size, sub_block_size)}")
    print(f"=== grid : {grid}")
    triton_mod_5d[grid](output, x, y, block_size, sub_block_size, *list(shape), *list(x.stride()))

    test_common.validate_cmp(dtype, ans, output)
