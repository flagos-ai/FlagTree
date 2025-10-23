import math
import pytest
import torch
import triton
import triton.language as tl
import test_common


@pytest.mark.parametrize("dtype", ['float32', 'float16', 'bfloat16', 'int32', 'int64', 'int16', 'int8'])
@pytest.mark.parametrize("M_BLOCK,N_BLOCK", [(2, 16), (8, 16)])
def test_tensor_descriptor_load_store(dtype, M_BLOCK, N_BLOCK):
    @triton.jit
    def kernel(out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
        in_desc = tl.make_tensor_descriptor(
            a_ptr,
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

    M, N = M_BLOCK * 2, N_BLOCK * 2
    inp = test_common.generate_tensor((M, N), dtype).npu()
    out = inp.new_empty((M, N))

    grid_m = M // M_BLOCK
    grid_n = N // N_BLOCK

    kernel[(grid_m, grid_n)](out, inp, M, N, M_BLOCK, N_BLOCK)
    torch.testing.assert_close(inp, out)


@pytest.mark.parametrize("dtype", ['float32', 'float16', 'bfloat16', 'int32', 'int64', 'int16', 'int8'])
def test_tensor_descriptor_load_store3d(dtype):

    @triton.jit
    def kernel(out_ptr, a_ptr, M, N, K, stride_m, stride_n, stride_k, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr,
               K_BLOCK: tl.constexpr):
        in_desc = tl.make_tensor_descriptor(
            a_ptr,
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

    M, N, K = 8, 16, 32
    inp = test_common.generate_tensor((M, N, K), dtype).npu()
    out = inp.new_empty((M, N, K))

    M_BLOCK = 2
    N_BLOCK = 4

    # 自动调整 K_BLOCK，保证最后一维 block 至少 16 字节
    dtype = getattr(inp, "dtype", None)
    itemsize = torch.tensor([], dtype=inp.dtype).element_size()
    min_k_block = max(16 // itemsize, 1)
    K_BLOCK = max(8, min_k_block)

    grid_m = M // M_BLOCK
    grid_n = N // N_BLOCK
    grid_k = K // K_BLOCK

    kernel[(grid_m, grid_n, grid_k)](out, inp, *inp.shape, *out.stride(), M_BLOCK, N_BLOCK, K_BLOCK)
    torch.testing.assert_close(inp.reshape(M * N * K), out.reshape(M * N * K))


# Exercise the functional load/store builtins once to ensure they map through.
@pytest.mark.parametrize("dtype", ["float32"])
def test_tensor_descriptor_functional_interface(dtype):
    """Copies an entire tensor blockwise using the descriptor builtins."""

    @triton.jit
    def kernel(out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
        in_desc = tl.make_tensor_descriptor(
            a_ptr,
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

    M, N = 32, 128
    inp = test_common.generate_tensor((M, N), dtype).npu()

    M_BLOCK = 8
    N_BLOCK = 32
    out = inp.new_empty((M, N))

    grid_m = M // M_BLOCK
    grid_n = N // N_BLOCK

    kernel[(grid_m, grid_n)](out, inp, M, N, M_BLOCK, N_BLOCK)
    torch.testing.assert_close(inp, out)


@pytest.mark.parametrize("dtype_str", ["int32"])
@pytest.mark.parametrize("shape", [(128, 2, 4), (64, 2, 4), (32, 2, 4), (2, 4, 32), (2, 4, 2)])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("device", ["npu"])
def test_reduce_max(dtype_str, shape, axis, device):

    @triton.jit
    def kernel(
        In,
        Out,
        in_shape1: tl.constexpr,
        in_shape2: tl.constexpr,
        in_shape3: tl.constexpr,
        ou_shape1: tl.constexpr,
        ou_shape2: tl.constexpr,
        axis: tl.constexpr,
    ):
        in_desc = tl.make_tensor_descriptor(
            base=In,
            shape=[in_shape1 * in_shape2 * in_shape3],
            strides=[1],
            block_shape=[in_shape1 * in_shape2 * in_shape3],
        )
        out_desc = tl.make_tensor_descriptor(
            base=Out,
            shape=[ou_shape1 * ou_shape2],
            strides=[1],
            block_shape=[ou_shape1 * ou_shape2],
        )
        val = in_desc.load([0]).reshape(in_shape1, in_shape2, in_shape3)
        output = tl.max(val, axis=axis)
        out_desc.store([0], output.reshape(out_desc.block_shape))

    inp = torch.arange(math.prod(shape), 
                         dtype=getattr(torch, dtype_str),
                         device=device).reshape(shape)
    expected, indices = torch.max(inp.to(torch.int64), dim=axis)
    expected = expected.to(torch.int32)
    actual = torch.zeros(expected.shape, dtype=getattr(torch, dtype_str), device=device)
    kernel[(1, )](inp, actual, *shape, *expected.shape, axis=axis)
    assert torch.equal(expected, actual)