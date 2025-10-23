import triton
import triton.language as tl
import pytest
import test_common
import torch
import torch_npu
import numpy as np


@triton.jit
def triton_test_fn_atomic_max_dma(
    in_ptr0, out_ptr0, out_ptr1, n_elements, BLOCK_SIZE: tl.constexpr
):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    tmp0 = tl.load(in_ptr0 + (x0))
    # only set mask of atomic_max
    tl.atomic_max(out_ptr0 + (x1), tmp0, xmask)


@triton.jit
def triton_test_fn_atomic_max_dma_supply(
    in_ptr0, out_ptr0, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.atomic_max(out_ptr0 + (x1), tmp0, xmask)
# torch.max do not support int
@pytest.mark.parametrize('param_list',
                         [
                             ['int16', (32, 32), 2],
                             ['float16', (32, 32), 2],
                             ['float32', (32, 32), 2],
                             ['float32', (128, 128), 8],
                             ['float32', (32768, 16), 32],
                             ['int32', (32, 32), 2],
                             ['int32', (128, 128), 8],
                             ['int32', (32768, 16), 32],
                         ]
                         )
def test_atomic_max(param_list):
    dtype, shape, ncore = param_list
    block_size = shape[0] * shape[1] / ncore
    split_size = shape[0] // ncore
    # old size: (32768, 256)
    # tensor of (1024, 256) is too big, and it will lead to failure in the backend
    # so here we make it smaller
    x0 = test_common.generate_tensor(shape, dtype)
    x1 = test_common.generate_tensor((split_size, shape[1]), dtype)
    y = test_common.generate_tensor((split_size, shape[1]), dtype)

    merged_tensor = torch.cat((x0, x1), dim=0)
    chunks = torch.stack(torch.chunk(merged_tensor, ncore+1, dim=0))
    x1_ref = torch.max(chunks, dim=0)[0]
    x0 = x0.npu()
    x1 = x1.npu()
    y = y.npu()

    n_elements = shape[0] * shape[1]
    triton_test_fn_atomic_max_dma[ncore, 1, 1](x0, x1, y, n_elements, BLOCK_SIZE=split_size * shape[1])
    test_common.validate_cmp(dtype, x1, x1_ref)


@pytest.mark.parametrize('invalid_param_list',
                         [
                             ['int64', (32, 32), 2],
                         ]
                         )
@test_common.raises_with_match(triton.compiler.errors.CompilationError, "not support int64")
def test_atomic_max_invalid(invalid_param_list):
    dtype, shape, ncore = invalid_param_list
    block_size = shape[0] * shape[1] / ncore
    split_size = shape[0] // ncore
    x0 = test_common.generate_tensor(shape, dtype)
    x1 = test_common.generate_tensor((split_size, shape[1]), dtype)
    y = test_common.generate_tensor((split_size, shape[1]), dtype)

    merged_tensor = torch.cat((x0, x1), dim=0)
    chunks = torch.stack(torch.chunk(merged_tensor, ncore+1, dim=0))
    x1_ref = torch.max(chunks, dim=0)[0]
    x0 = x0.npu()
    x1 = x1.npu()
    y = y.npu()

    n_elements = shape[0] * shape[1]
    triton_test_fn_atomic_max_dma[ncore, 1, 1](x0, x1, y, n_elements, BLOCK_SIZE=split_size * shape[1])
    test_common.validate_cmp(dtype, x1, x1_ref)


@pytest.mark.parametrize('shape', [(3, 1), (13, 1), (32, 1), (256, 1)])
@pytest.mark.parametrize('dtype', ['float32'])
def test_atomic_max_2d_supply(dtype, shape):
    # old size: (32768, 256)
    # tensor of (1024, 256) is too big, and it will lead to failure in the backend
    # so here we make it smaller
    x0 = test_common.generate_tensor(shape, dtype).npu()
    x1 = test_common.generate_tensor(shape, dtype).npu()

    x1_ref = torch.maximum(x0, x1)

    n_elements = shape[0] * shape[1]
    triton_test_fn_atomic_max_dma_supply[shape[0], 1, 1](x0, x1, n_elements, BLOCK_SIZE=shape[1])
    test_common.validate_cmp(dtype, x1, x1_ref)
# if __name__ == "__main__":
#     test_atomic_max(['int32', (8, 8), 2])