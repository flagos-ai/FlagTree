import logging
import triton
import torch
import pytest
import test_common

import numpy as np
import triton.language as tl
from test_common import TestUtils

# inp the two 32 bit signed integers.
@triton.jit
def umulhi_kernel(X, Y, Z, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(X + offs)
    y = tl.load(Y + offs)
    z = tl.umulhi(x, y)
    tl.store(Z + tl.arange(0, N), z)


@triton.jit
def triton_umulhi_4d_5d(
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
    ret = tl.umulhi(x_val, y_val)
    tl.store(output_ptr + offsets, ret, mask=masks)


# accuracy reference
def umulhi32(a, b):
    a_64 = a.astype(np.int64)
    b_64 = b.astype(np.int64)
    product_64 = a_64 * b_64
    # get the high part
    result_high_32 = product_64 >> 32
    return result_high_32.astype(np.int32)


@pytest.mark.parametrize('dtype', ['int32'])
@pytest.mark.parametrize('shape', TestUtils.full_shape)
def test_case2(dtype, shape):
    N = shape[0]
    dtypes = eval('torch.' + dtype)
    x = torch.randint(low=0, high=2000, size=shape, dtype=dtypes)
    y = torch.randint(low=0, high=2000, size=shape, dtype=dtypes)
    xx = x.npu()
    yy = y.npu()
    z_tri = torch.zeros(size=shape, dtype=dtypes).npu()
    umulhi_kernel[(1,)](xx, yy, z_tri, N=N)

    xxx = x.numpy()
    yyy = y.numpy()
    z_ref = umulhi32(xxx, yyy)
    z_ref1 = torch.from_numpy(z_ref).npu()
    torch.equal(z_tri, z_ref1)

invalid_types = [
    'int8',
    'int16',
    'int64',
    'float16',
    'float32',
    'bfloat16',
    'bool',
]
@pytest.mark.parametrize("dtype", invalid_types)
@test_common.raises_with_match(triton.compiler.errors.CompilationError, "Expected dtype")
def test_umulhi_invalid_dtype_case(dtype):
    x0 = test_common.generate_tensor((1,), dtype).npu()
    x1 = test_common.generate_tensor((1,), dtype).npu()

    y_cal = torch.zeros((1,), dtype=eval('torch.' + dtype)).npu()
    umulhi_kernel[(1,)](x0, x1, y_cal, 1)


@pytest.mark.parametrize('shape', TestUtils.test_shape4d + TestUtils.test_shape5d)
@pytest.mark.parametrize('dtype', ['int32'])
def test_umulhi_4d_5d(shape, dtype):
    logging.log(logging.DEBUG, f"shape = {shape}")
    x = torch.randint(low=0, high=2000, size=shape, dtype=eval('torch.' + dtype))
    y = torch.randint(low=0, high=2000, size=shape, dtype=eval('torch.' + dtype))
    xx = x.npu()
    yy = y.npu()

    output = torch.zeros(size=shape, dtype=eval('torch.' + dtype)).npu()

    logging.log(logging.DEBUG, f"output.dtype={output.dtype}")

    xxx = x.numpy()
    yyy = y.numpy()
    z = umulhi32(xxx, yyy)
    ans = torch.from_numpy(z).npu()

    blocks = list(x.size())
    strides = list(x.stride())
    while len(blocks) < 5:
        blocks.append(1)
        strides.append(1)

    grid = (1,)
    triton_umulhi_4d_5d[grid](output, xx, yy, *blocks, *blocks, *strides)

    test_common.validate_cmp(dtype, ans, output)