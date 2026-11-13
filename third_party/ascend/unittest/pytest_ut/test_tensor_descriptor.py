# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import math
import pytest
import torch
import triton
import triton.language as tl
import test_common


@pytest.mark.parametrize("dtype", ['float32', 'int32'])
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


@pytest.mark.parametrize("dtype", ['float32', 'int32'])
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
@pytest.mark.parametrize("dtype", ["float32", "uint8"])
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
@pytest.mark.parametrize("shape", [(128, 2, 4), (2, 4, 2)])
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

    inp = torch.arange(math.prod(shape), dtype=getattr(torch, dtype_str), device=device).reshape(shape)
    expected, indices = torch.max(inp.to(torch.int64), dim=axis)
    expected = expected.to(torch.int32)
    actual = torch.zeros(expected.shape, dtype=getattr(torch, dtype_str), device=device)
    kernel[(1, )](inp, actual, *shape, *expected.shape, axis=axis)
    assert torch.equal(expected, actual)


@pytest.mark.parametrize("dtype", ['float32', 'float16', 'bfloat16'])
@pytest.mark.parametrize("padding", ["zero", "nan"])
def test_tensor_descriptor_padding(dtype, padding):

    @triton.jit
    def device_tma_load(in_ptr, out_ptr, IM, IN, YM, YN, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr,
                        padding: tl.constexpr):
        x_desc = tl.make_tensor_descriptor(in_ptr, shape=[IM, IN], strides=[IN, 1], block_shape=[M_BLOCK, N_BLOCK],
                                           padding_option=padding)
        out_desc = tl.make_tensor_descriptor(out_ptr, shape=[YM, YN], strides=[YN, 1], block_shape=[M_BLOCK, N_BLOCK])

        moffset = tl.program_id(0) * M_BLOCK
        noffset = tl.program_id(1) * N_BLOCK

        value = x_desc.load([moffset, noffset])

        out_desc.store([moffset, noffset], value)

    def alloc_fn(size: int, alignment: float, stream: float):
        return torch.ones(size, device="npu", dtype=torch.float32)

    triton.set_allocator(alloc_fn)

    IM, IN = 48, 48
    OM, ON = 64, 64
    M_BLOCK = 32
    N_BLOCK = 32
    torch_dtype = getattr(torch, dtype)
    input_tensor = torch.arange(IM * IN, device="npu", dtype=torch_dtype).reshape(IM, IN)
    out_device_tma = torch.zeros((OM, ON), device="npu", dtype=torch_dtype)
    grid = (triton.cdiv(OM, M_BLOCK), triton.cdiv(ON, N_BLOCK))
    device_tma_load[grid](input_tensor, out_device_tma, IM, IN, OM, ON, M_BLOCK, N_BLOCK, padding)
    expected = torch.zeros((OM, ON), device="npu", dtype=torch_dtype)
    expected[0:IN, 0:IM] = input_tensor
    if padding == "nan":
        expected[:, IN:ON] = float('nan')
        expected[IM:OM, :] = float('nan')

    torch.testing.assert_close(expected, out_device_tma, equal_nan=True)


@pytest.mark.parametrize("X, Y", [(128, 128)])
@pytest.mark.parametrize("BLOCK_X, BLOCK_Y", [(32, 32)])
@pytest.mark.parametrize("dtype", ['float32', 'int32'])
@pytest.mark.parametrize("y", [0, 32])
def test_tensor_descriptor_scatter(X, Y, BLOCK_X, BLOCK_Y, dtype, y):

    def torch_scatter_rows(input, idx, y, block_y, X, Y):
        out = torch.zeros((X, Y), dtype=input.dtype, device=input.device)
        for i, j in enumerate(idx):
            out[j][y:y + block_y] = input[i]
        return out

    @triton.jit
    def tensor_descriptor_scatter_rows_kernel(out_ptr, in_ptr, idx_ptr, y, X: tl.constexpr, Y: tl.constexpr,
                                              BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr):
        idx = tl.load(idx_ptr + tl.arange(0, BLOCK_X))
        data = tl.load(in_ptr + tl.arange(0, BLOCK_X)[:, None] * BLOCK_Y + tl.arange(0, BLOCK_Y)[None, :])
        desc = tl.make_tensor_descriptor(out_ptr, [X, Y], [Y, 1], [1, BLOCK_Y])
        desc.scatter(data, idx, y)

    device = 'npu'
    if BLOCK_X > X or y + BLOCK_Y > Y:
        pytest.skip()

    torch.manual_seed(42)
    torch_dtype = getattr(torch, dtype)
    input_tensor = torch.arange(BLOCK_X * BLOCK_Y, dtype=torch_dtype, device=device).reshape(BLOCK_X, BLOCK_Y)
    output = torch.zeros((X, Y), dtype=torch_dtype, device=device)

    idx = torch.randperm(BLOCK_X, dtype=torch.int32, device=device)

    def alloc_fn(size: int, align: int, stream):
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)

    tensor_descriptor_scatter_rows_kernel[(1, )](output, input_tensor, idx, y, X, Y, BLOCK_X, BLOCK_Y)

    ref = torch_scatter_rows(input_tensor, idx, y, BLOCK_Y, X, Y)
    torch.testing.assert_close(ref, output, atol=0, rtol=0)


@pytest.mark.parametrize("X, Y", [(128, 128)])
@pytest.mark.parametrize("BLOCK_X, BLOCK_Y", [(32, 32)])
@pytest.mark.parametrize("dtype", ['float32', 'int32'])
@pytest.mark.parametrize("y", [0, 32])
def test_tensor_descriptor_gather(X, Y, BLOCK_X, BLOCK_Y, dtype, y):

    @triton.jit
    def tensor_descriptor_gather_rows_kernel(out_ptr, in_ptr, idx_ptr, y, X: tl.constexpr, Y: tl.constexpr,
                                             BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr):
        idx = tl.load(idx_ptr + tl.arange(0, BLOCK_X))
        desc = tl.make_tensor_descriptor(in_ptr, [X, Y], [Y, 1], [1, BLOCK_Y])
        out = desc.gather(idx, y)
        tl.store(out_ptr + tl.arange(0, BLOCK_X)[:, None] * BLOCK_Y + tl.arange(0, BLOCK_Y)[None, :], out)

    def torch_gather_rows(input, idx, y, block_y):
        return input[idx.long(), y:y + block_y]

    device = 'npu'
    if BLOCK_X > X or y + BLOCK_Y > Y:
        pytest.skip()

    torch.manual_seed(42)
    torch_dtype = getattr(torch, dtype)
    input_tensor = test_common.generate_tensor((X, Y), dtype).npu()
    output = torch.empty((BLOCK_X, BLOCK_Y), dtype=torch_dtype, device=device)

    idx = torch.randint(BLOCK_X, (BLOCK_X, ), dtype=torch.int32, device=device)

    def alloc_fn(size: int, align: int, steam):
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)

    tensor_descriptor_gather_rows_kernel[(1, )](output, input_tensor, idx, y, X, Y, BLOCK_X, BLOCK_Y)

    ref = torch_gather_rows(input_tensor, idx, y, BLOCK_Y)
    torch.testing.assert_close(ref, output, atol=0, rtol=0)


REDUCE_OP = {
    "add": lambda a, b: a + b,
    "min": lambda a, b: torch.minimum(a, b),
    "max": lambda a, b: torch.maximum(a, b),
    "and": lambda a, b: torch.bitwise_and(a, b),
    "or": lambda a, b: torch.bitwise_or(a, b),
    "xor": lambda a, b: torch.bitwise_xor(a, b),
}

SKIP_KINDS = {"and", "or", "xor"}
all_kinds = ["add", "min", "max", "and", "or", "xor"]
kind_parms = [
    pytest.param(k, marks=pytest.mark.skip(
        reason=f"skip for bishengir compile failed on a2,succeed on a5")) if k in SKIP_KINDS else k for k in all_kinds
]


@pytest.mark.parametrize("kind", kind_parms)
@pytest.mark.parametrize("dtype", ['int32'])
@pytest.mark.parametrize("M_BLOCK,N_BLOCK", [(2, 16)])
def test_tensor_descriptor_reduce(kind, dtype, M_BLOCK, N_BLOCK):

    @triton.jit(debug=True)
    def kernel(a_ptr, out_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr, kind: tl.constexpr):
        moffset = tl.program_id(0) * M_BLOCK
        noffset = tl.program_id(1) * N_BLOCK

        midx = moffset + tl.arange(0, M_BLOCK)[:, None]
        nidx = noffset + tl.arange(0, N_BLOCK)[None, :]
        idx = midx * N + nidx
        val = tl.load(a_ptr + idx)

        desc = tl.make_tensor_descriptor(
            out_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[M_BLOCK, N_BLOCK],
        )

        if kind == "add":
            desc.atomic_add([moffset, noffset], val)
        elif kind == "min":
            desc.atomic_min([moffset, noffset], val)
        elif kind == "max":
            desc.atomic_max([moffset, noffset], val)
        elif kind == "and":
            desc.atomic_and([moffset, noffset], val)
        elif kind == "or":
            desc.atomic_or([moffset, noffset], val)
        else:
            tl.static_assert(kind == "xor")
            desc.atomic_xor([moffset, noffset], val)

    M, N = M_BLOCK * 2, N_BLOCK * 2
    inp = test_common.generate_tensor((M, N), dtype).npu()
    out = test_common.generate_tensor((M, N), dtype).npu()

    grid_m = M // M_BLOCK
    grid_n = N // N_BLOCK

    expect = REDUCE_OP[kind](inp, out)
    kernel[(grid_m, grid_n)](inp, out, M, N, M_BLOCK, N_BLOCK, kind)
    torch.testing.assert_close(expect, out)


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("M,N,BLOCK_M,BLOCK_N", [(64, 128, 16, 32)])
def test_host_tensor_descriptor_args(dtype, M, N, BLOCK_M, BLOCK_N):
    from triton.tools.tensor_descriptor import TensorDescriptor

    @triton.jit
    def kernel(in_desc, out_desc, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid = tl.program_id(0)
        grid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        off_m = (pid_m * BLOCK_M).to(tl.int32)
        off_n = (pid_n * BLOCK_N).to(tl.int32)
        tile = in_desc.load([off_m, off_n])
        out_desc.store([off_m, off_n], tile)

    torch_dtype = getattr(torch, dtype)
    inp = torch.randn((M, N), dtype=torch_dtype, device="npu")
    out = torch.empty((M, N), dtype=torch_dtype, device="npu")
    in_desc = TensorDescriptor(inp, list(inp.shape), list(inp.stride()), [BLOCK_M, BLOCK_N])
    out_desc = TensorDescriptor(out, list(out.shape), list(out.stride()), [BLOCK_M, BLOCK_N])

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    kernel[grid](in_desc, out_desc, M, N, BLOCK_M, BLOCK_N)
    torch.testing.assert_close(inp, out)


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
@pytest.mark.parametrize("padding", ["zero", "nan"])
def test_host_tensor_descriptor_padding(padding):
    from triton.tools.tensor_descriptor import TensorDescriptor

    @triton.jit
    def kernel(in_desc, out_desc, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        moffset = (tl.program_id(0) * BLOCK_M).to(tl.int32)
        noffset = (tl.program_id(1) * BLOCK_N).to(tl.int32)
        value = in_desc.load([moffset, noffset])
        out_desc.store([moffset, noffset], value)

    IM, IN = 48, 48
    OM, ON = 64, 64
    BLOCK_M, BLOCK_N = 32, 32
    inp = torch.arange(IM * IN, device="npu", dtype=torch.float32).reshape(IM, IN)
    out = torch.zeros((OM, ON), device="npu", dtype=torch.float32)

    in_desc = TensorDescriptor(inp, list(inp.shape), list(inp.stride()), [BLOCK_M, BLOCK_N], padding=padding)
    out_desc = TensorDescriptor(out, list(out.shape), list(out.stride()), [BLOCK_M, BLOCK_N])

    grid = (triton.cdiv(OM, BLOCK_M), triton.cdiv(ON, BLOCK_N))
    kernel[grid](in_desc, out_desc, BLOCK_M, BLOCK_N)

    expected = torch.zeros((OM, ON), device="npu", dtype=torch.float32)
    expected[0:IM, 0:IN] = inp
    if padding == "nan":
        expected[IM:OM, :] = float("nan")
        expected[:, IN:ON] = float("nan")
        # corner already nan from both; keep as nan
        expected[IM:OM, IN:ON] = float("nan")
    torch.testing.assert_close(expected, out, equal_nan=True)
