import triton
import triton.language as tl
import torch
import pytest
import test_common
from test_common import TestUtils
import math
import numpy as np
import scipy


@triton.jit
def kernel_rand(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(block_size):
        global_offset = block_offset + inner_idx
        rand_vals = tl.rand(5, 10 + global_offset, n_rounds)  # 对每个索引生成一个随机数
        tl.store(x_ptr + global_offset, rand_vals)  # 存储随机数


@triton.jit
def triton_rand_4d_5d(
        output_ptr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr,
        BLOCK_3: tl.constexpr, BLOCK_4: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr,
        SHAPE_3: tl.constexpr, SHAPE_4: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr,
        STRIDE_3: tl.constexpr, STRIDE_4: tl.constexpr
):
    # 1D program_id for flatten multi-d offset
    pid = tl.program_id(0)
    # base offset for dimension 0
    offsets = pid + tl.arange(0, BLOCK_0) * STRIDE_0
    mask = tl.arange(0, BLOCK_0) < SHAPE_0
    # nested offset expansion
    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        mask = mask[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        mask = mask[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        mask = mask[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if BLOCK_4 > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        mask = mask[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)

    ret = tl.rand(5, offsets, 10)
    tl.store(output_ptr + offsets, ret, mask=mask)


@triton.jit
def kernel_randn(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(block_size):
        global_offset = block_offset + inner_idx
        rand_vals = tl.randn(5, 10 + global_offset, n_rounds)  # 对每个索引生成一个随机数
        tl.store(x_ptr + global_offset, rand_vals)  # 存储随机数


@triton.jit
def triton_randn_4d_5d(
        output_ptr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr,
        BLOCK_3: tl.constexpr, BLOCK_4: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr,
        SHAPE_3: tl.constexpr, SHAPE_4: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr,
        STRIDE_3: tl.constexpr, STRIDE_4: tl.constexpr
):
    # 1D program_id for flatten multi-d offset
    pid = tl.program_id(0)
    # base offset for dimension 0
    offsets = pid + tl.arange(0, BLOCK_0) * STRIDE_0
    mask = tl.arange(0, BLOCK_0) < SHAPE_0
    # nested offset expansion
    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        mask = mask[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        mask = mask[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        mask = mask[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if BLOCK_4 > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        mask = mask[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)

    ret = tl.randn(5, offsets, 10)
    tl.store(output_ptr + offsets, ret, mask=mask)


@triton.jit
def kernel_randint(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(block_size):
        global_offset = block_offset + inner_idx
        rand_vals = tl.randint(5, 10 + global_offset, n_rounds)  # 对每个索引生成一个随机数
        tl.store(x_ptr + global_offset, rand_vals)  # 存储随机数


@triton.jit
def triton_randint_4d_5d(
        output_ptr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr,
        BLOCK_3: tl.constexpr, BLOCK_4: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr,
        SHAPE_3: tl.constexpr, SHAPE_4: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr,
        STRIDE_3: tl.constexpr, STRIDE_4: tl.constexpr
):
    # 1D program_id for flatten multi-d offset
    pid = tl.program_id(0)
    # base offset for dimension 0
    offsets = pid + tl.arange(0, BLOCK_0) * STRIDE_0
    mask = tl.arange(0, BLOCK_0) < SHAPE_0
    # nested offset expansion
    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        mask = mask[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        mask = mask[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        mask = mask[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if BLOCK_4 > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        mask = mask[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)

    ret = tl.randint(5, offsets, 10)
    tl.store(output_ptr + offsets, ret, mask=mask)


@triton.jit
def kernel_randint4x(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    indices = tl.arange(0, 4)
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(0, block_size + 4, step=4):
        global_offset = block_offset + inner_idx
        rand_vals = tl.randint4x(5, 10 + global_offset, n_rounds)  # 对每个索引生成一个随机数
        mask = (global_offset + indices) < (block_offset + block_size)
        tl.store(x_ptr + global_offset + indices, rand_vals, mask)  # 存储随机数


@triton.jit
def triton_randint4x_4d_5d(
        output_ptr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr,
        BLOCK_3: tl.constexpr, BLOCK_4: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr,
        SHAPE_3: tl.constexpr, SHAPE_4: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr,
        STRIDE_3: tl.constexpr, STRIDE_4: tl.constexpr
):
    # 1D program_id for flatten multi-d offset
    pid = tl.program_id(0)
    # base offset for dimension 0
    offsets = pid + tl.arange(0, BLOCK_0) * STRIDE_0
    mask = tl.arange(0, BLOCK_0) < SHAPE_0
    # nested offset expansion
    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        mask = mask[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        mask = mask[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        mask = mask[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if BLOCK_4 > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        mask = mask[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)

    ret = tl.randint4x(5, offsets, 10)
    tl.store(output_ptr + offsets, ret, mask=mask)


# With alpha=0.01, z=-3.0902, N=100, we have (1-0.01)+(-3.0902)*sqrt(0.01*(1-0.01)/100)=0.9593,
# so there must be 96 cases for each shape to have pvalue larger than 0.01.
# There is higher possibility to fail with small shapes, so we will use large shape.
@pytest.mark.parametrize('shape', [
    (256, 256),
    (512, 512),
    (1024, 1024),
])
def test_rand_case(shape):
    y_calf = torch.zeros(shape, dtype=eval('torch.float32')).npu()

    numel = y_calf.numel()
    ncore = 1 if numel < 32 else 32
    xblock = math.ceil(numel / ncore)

    correctness = 0
    for _ in range(100):
        ref = np.random.random_sample(shape).flatten()
        kernel_rand[ncore, 1, 1](y_calf, 10, numel, xblock)
        
        pvalue = scipy.stats.kstest(ref, y_calf.cpu().numpy().flatten()).pvalue
        if pvalue > 0.01:
            correctness += 1

    assert correctness > 95


@pytest.mark.parametrize('shape', [
    (256, 256),
    (512, 512),
    (1024, 1024),
])
def test_randn_case(shape):
    y_calf = torch.zeros(shape, dtype=eval('torch.float32')).npu()

    numel = y_calf.numel()
    ncore = 1 if numel < 32 else 32
    xblock = math.ceil(numel / ncore)

    correctness = 0
    for _ in range(100):
        ref = np.random.standard_normal(shape).flatten()
        kernel_randn[ncore, 1, 1](y_calf, 10, numel, xblock)
        
        pvalue = scipy.stats.kstest(ref, y_calf.cpu().numpy().flatten()).pvalue
        if pvalue > 0.01:
            correctness += 1

    assert correctness > 95


@pytest.mark.parametrize('shape', [
    (256, 256),
    (512, 512),
    (1024, 1024),
])
def test_randint_case(shape):
    y_cali = torch.zeros(shape, dtype=eval('torch.int32')).npu()

    numel = y_cali.numel()
    ncore = 1 if numel < 32 else 32
    xblock = math.ceil(numel / ncore)

    correctness = 0
    ii32 = np.iinfo(np.int32)
    for _ in range(100):
        ref = np.random.randint(low=ii32.min, high=ii32.max, size=shape).flatten()
        kernel_randint[ncore, 1, 1](y_cali, 10, numel, xblock)
        
        pvalue = scipy.stats.kstest(ref, y_cali.cpu().numpy().flatten()).pvalue
        if pvalue > 0.01:
            correctness += 1

    assert correctness > 95


@pytest.mark.parametrize('shape', [
    (256, 256),
    (512, 512),
    (1024, 1024),
])
def test_randint4x_case(shape):
    y_cali = torch.zeros(shape, dtype=eval('torch.int32')).npu()

    numel = y_cali.numel()
    ncore = 1 if numel < 32 else 32
    xblock = math.ceil(numel / ncore)

    correctness = 0
    ii32 = np.iinfo(np.int32)
    for _ in range(100):
        ref = np.random.randint(low=ii32.min, high=ii32.max, size=shape).flatten()
        kernel_randint4x[ncore, 1, 1](y_cali, 10, numel, xblock)
        
        pvalue = scipy.stats.kstest(ref, y_cali.cpu().numpy().flatten()).pvalue
        if pvalue > 0.01:
            correctness += 1

    assert correctness > 95


@pytest.mark.parametrize('shape', TestUtils.test_shape4d + TestUtils.test_shape5d)
def test_rand_4d_5d(shape):
    x = torch.zeros(shape, dtype=eval('torch.float32')).npu()
    y = torch.zeros(shape, dtype=eval('torch.int32')).npu()

    blocks = list(x.size())
    strides = list(x.stride())
    while len(blocks) < 5:
        blocks.append(1)
        strides.append(1)

    grid = (1,)
    triton_rand_4d_5d[grid](x, *blocks, *blocks, *strides)
    triton_randn_4d_5d[grid](x, *blocks, *blocks, *strides)
    triton_randint_4d_5d[grid](y, *blocks, *blocks, *strides)
    triton_randint4x_4d_5d[grid](y, *blocks, *blocks, *strides)
