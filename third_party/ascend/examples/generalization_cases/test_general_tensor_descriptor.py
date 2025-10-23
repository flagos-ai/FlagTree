import triton
import triton.language as tl

import torch
import torch_npu
import pytest
import test_common
from test_common import TestUtils


@triton.jit
def triton_tensor_descriptor_2d(
        out_ptr, x_ptr,
        M: tl.constexpr, N: tl.constexpr,
        M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr,
):
    in_desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )
    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK
    block = in_desc.load([moffset, noffset])
    out_desc.store([moffset, noffset], block)


@triton.jit
def triton_tensor_descriptor_3d(
        out_ptr, x_ptr,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        stride_m: tl.constexpr, stride_n: tl.constexpr, stride_k: tl.constexpr,
        M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr, K_BLOCK: tl.constexpr,
):
    in_desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[M, N, K],
        strides=[stride_m, stride_n, stride_k],
        block_shape=[M_BLOCK, N_BLOCK, K_BLOCK],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[M, N, K],
        strides=[stride_m, stride_n, stride_k],
        block_shape=[M_BLOCK, N_BLOCK, K_BLOCK],
    )
    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK
    koffset = tl.program_id(2) * K_BLOCK
    block = in_desc.load([moffset, noffset, koffset])
    out_desc.store([moffset, noffset, koffset], block)


@triton.jit
def triton_tensor_descriptor_4d(
        out_ptr, x_ptr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr, 
        SHAPE_3: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr, 
        STRIDE_3: tl.constexpr, 
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr, 
        BLOCK_3: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)
    idx2 = pid2 // BLOCK_3
    idx3 = pid2 % BLOCK_3
    o1 = pid0 * BLOCK_0
    o2 = pid1 * BLOCK_1
    o3 = idx2 * BLOCK_2
    o4 = idx3 * BLOCK_3
    in_desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[SHAPE_0, SHAPE_1, SHAPE_2, SHAPE_3],
        strides=[STRIDE_0, STRIDE_1, STRIDE_2, STRIDE_3],
        block_shape=[BLOCK_0, BLOCK_1, BLOCK_2, BLOCK_3],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[SHAPE_0, SHAPE_1, SHAPE_2, SHAPE_3],
        strides=[STRIDE_0, STRIDE_1, STRIDE_2, STRIDE_3],
        block_shape=[BLOCK_0, BLOCK_1, BLOCK_2, BLOCK_3],
    )
    block = in_desc.load([o1, o2, o3, o4])
    out_desc.store([o1, o2, o3, o4], block)


@triton.jit
def triton_tensor_descriptor_5d(
        out_ptr, x_ptr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr,
        SHAPE_3: tl.constexpr, SHAPE_4: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr,
        STRIDE_3: tl.constexpr, STRIDE_4: tl.constexpr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr, 
        BLOCK_3: tl.constexpr, BLOCK_4: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)
    idx3 = pid2 // (BLOCK_3 * BLOCK_4)
    idx4 = (pid2 // BLOCK_4) % BLOCK_3
    idx5 = pid2 % BLOCK_4
    o1 = pid0 * BLOCK_0
    o2 = pid1 * BLOCK_1
    o3 = idx3 * BLOCK_2
    o4 = idx4 * BLOCK_3
    o5 = idx5 * BLOCK_4
    in_desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[SHAPE_0, SHAPE_1, SHAPE_2, SHAPE_3, SHAPE_4],
        strides=[STRIDE_0, STRIDE_1, STRIDE_2, STRIDE_3, STRIDE_4],
        block_shape=[BLOCK_0, BLOCK_1, BLOCK_2, BLOCK_3, BLOCK_4],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[SHAPE_0, SHAPE_1, SHAPE_2, SHAPE_3, SHAPE_4],
        strides=[STRIDE_0, STRIDE_1, STRIDE_2, STRIDE_3, STRIDE_4],
        block_shape=[BLOCK_0, BLOCK_1, BLOCK_2, BLOCK_3, BLOCK_4],
    )
    block = in_desc.load([o1, o2, o3, o4, o5])
    out_desc.store([o1, o2, o3, o4, o5], block)


@triton.jit
def triton_tensor_descriptor_function_2d(
        out_ptr, x_ptr,
        M: tl.constexpr, N: tl.constexpr,
        M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr,
):
    in_desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )
    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK
    block = tl.load_tensor_descriptor(in_desc, [moffset, noffset])
    tl.store_tensor_descriptor(out_desc, [moffset, noffset], block)


temporarily_not_support_dtype = ['bool']


@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
@pytest.mark.parametrize('shape', TestUtils.full_shape)
def test_tensor_descriptor_load_store_nd(dtype, shape):
    """测试tensor_descriptor的load和store功能"""

    if dtype in temporarily_not_support_dtype:
        pytest.skip(f"{dtype} not supported")

    inp = test_common.generate_tensor(shape, dtype).npu()
    out = inp.new_empty(shape)
    blocks = list(inp.size())
    strides = list(inp.stride())
    grid = (1,)
    dims = len(shape)

    # 如果最后一维小于16字节，则跳过
    itemsize = torch.tensor([], dtype=inp.dtype).element_size()
    if blocks[-1] * itemsize < 16:
        pytest.skip(f"last dimension must be at least 16 bytes, but got {blocks[-1] * itemsize} bytes")

    if dims == 2:
        if inp.numel() * inp.element_size() > 8192:
            triton_tensor_descriptor_2d[shape[0], 1, 1](out, inp, 1, shape[1], 1, shape[1])
        else:
            triton_tensor_descriptor_2d[grid](out, inp, *shape, *blocks)
        torch.testing.assert_close(inp, out)
    elif dims == 3:
        triton_tensor_descriptor_3d[grid](out, inp, *shape, *strides, *blocks)
        torch.testing.assert_close(inp, out)
    elif dims == 4:
        triton_tensor_descriptor_4d[grid](out, inp, *shape, *strides, *blocks)
        torch.testing.assert_close(inp, out)
    elif dims == 5:
        triton_tensor_descriptor_5d[grid](out, inp, *shape, *strides, *blocks)
        torch.testing.assert_close(inp, out)
    else:
        pytest.skip(f"{dims}d not supported")


@pytest.mark.parametrize("dtype", ["float32"])
def test_tensor_descriptor_in_function(dtype):
    """测试函数式接口是否正常工作"""
    
    M, N = 32, 128
    inp = test_common.generate_tensor((M, N), dtype).npu()
    out = inp.new_empty((M, N))

    M_BLOCK = 8
    N_BLOCK = 32
    grid_m = M // M_BLOCK
    grid_n = N // N_BLOCK

    triton_tensor_descriptor_function_2d[(grid_m, grid_n)](out, inp, M, N, M_BLOCK, N_BLOCK)
    torch.testing.assert_close(inp, out)
