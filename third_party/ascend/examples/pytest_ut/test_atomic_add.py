import triton
import triton.language as tl
import pytest
import test_common
import torch
import torch_npu

@triton.jit
def atomic_add(in_ptr0, out_ptr0, out_ptr1, n_elements, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.atomic_add(out_ptr0 + (x1), tmp0, xmask)
    tl.store(out_ptr1 + (x1), tmp1, xmask)


@triton.jit
def atomic_add_supply(
    in_ptr0, out_ptr0, n_elements, BLOCK_SIZE: tl.constexpr
):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.atomic_add(out_ptr0 + (x1), tmp0, xmask)
@pytest.mark.parametrize('param_list',
                         [
                             ['int16', (32, 32), 2],
                             ['int8', (32, 32), 2],
                             ['float32', (32, 32), 2],
                             ['float16', (64, 64), 4],
                             ['float32', (128, 128), 8],
                             ['float16', (128, 128), 16],
                             ['float32', (32768, 16), 32],
                         ]
                         )
def test_atomic_add(param_list):
    dtype, shape, ncore = param_list
    block_size = shape[0] * shape[1] / ncore
    split_size = shape[0] // ncore
    x0_value = 3
    x0 = torch.full(shape, x0_value, dtype = eval(f'torch.{dtype}')).npu()
    x1 = torch.full((split_size, shape[1]), 2, dtype = eval(f'torch.{dtype}')).npu()
    y = torch.full((split_size, shape[1]), -10, dtype = eval(f'torch.{dtype}')).npu()

    y_ref = x1 + 0
    x1_ref = x1 + ncore * x0_value

    n_elements = shape[0] * shape[1]
    atomic_add[ncore, 1, 1](x0, x1, y, n_elements, BLOCK_SIZE=split_size * shape[1])
    test_common.validate_cmp(dtype, x1, x1_ref)

@pytest.mark.parametrize('invalid_param_list',
                         [
                             ['int64', (32, 32), 2],
                         ]
                         )
@test_common.raises_with_match(triton.compiler.errors.CompilationError, "not support int64")
def test_atomic_add_invalid(invalid_param_list):
    dtype, shape, ncore = invalid_param_list
    block_size = shape[0] * shape[1] / ncore
    split_size = shape[0] // ncore
    x0_value = 3
    x0 = torch.full(shape, x0_value, dtype = eval(f'torch.{dtype}')).npu()
    x1 = torch.full((split_size, shape[1]), 2, dtype = eval(f'torch.{dtype}')).npu()
    y = torch.full((split_size, shape[1]), -10, dtype = eval(f'torch.{dtype}')).npu()
    y_ref = x1 + 0
    x1_ref = x1 + ncore * x0_value
    n_elements = shape[0] * shape[1]
    atomic_add[ncore, 1, 1](x0, x1, y, n_elements, BLOCK_SIZE=split_size * shape[1])
    test_common.validate_cmp(dtype, x1, x1_ref)

@triton.jit
def atomic_add_2d(in_ptr0, out_ptr0, out_ptr1, numel_0, numel_1, BLOCK_SIZE_0 : tl.constexpr, BLOCK_SIZE_1 : tl.constexpr):
    pid = tl.program_id(0)
    idx0_in = pid * BLOCK_SIZE_0 + tl.arange(0, BLOCK_SIZE_0)[:, None]
    idx0_out = tl.arange(0, BLOCK_SIZE_0)[:, None]
    idx1 = tl.arange(0, BLOCK_SIZE_1)[None, :]
    idx_in = idx0_in * BLOCK_SIZE_1 + idx1
    idx_out = idx0_out * BLOCK_SIZE_1 + idx1
    msk_in = (idx0_in < numel_0) & (idx1 < numel_1)
    msk_out = (idx0_out < numel_0) & (idx1 < numel_1)
    tmp0 = tl.load(in_ptr0 + idx_in, msk_in)
    tmp1 = tl.atomic_add(out_ptr0 + idx_out, tmp0, msk_out)
    tl.store(out_ptr1 + idx_out, tmp1, msk_out)

@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (32, 32), 2],
                         ]
                         )
def test_atomic_add_2d(param_list):
    dtype, shape, ncore = param_list
    split_size = shape[0] // ncore
    block_size_0 = split_size
    block_size_1 = shape[1]
    x0_value = 3
    x0 = torch.full(shape, x0_value, dtype = eval('torch.float32')).npu()
    x1 = torch.full((split_size, shape[1]), 2, dtype = eval('torch.float32')).npu()
    y = torch.full((split_size, shape[1]), -10, dtype = eval('torch.float32')).npu()

    y_ref = x1 + 0
    x1_ref = x1 + ncore * x0_value

    atomic_add_2d[ncore, 1, 1](x0, x1, y, shape[0], shape[1], BLOCK_SIZE_0=block_size_0, BLOCK_SIZE_1=block_size_1)
    test_common.validate_cmp(dtype, x1, x1_ref)


@pytest.mark.parametrize('shape', [(3, 1), (13, 1), (32, 1), (256, 1)])
@pytest.mark.parametrize('dtype', ['float32'])
def test_atomic_add_2d_supply(dtype, shape):
    ncore = 1
    block_size = shape[0] * shape[1] / ncore
    split_size = shape[0] // ncore
    x0_value = 3
    x0 = torch.full(shape, x0_value, dtype=eval('torch.' + dtype)).npu()
    x1 = torch.full((split_size, shape[1]), 2, dtype=eval('torch.' + dtype)).npu()

    y_ref = x1 + 0
    x1_ref = x1 + ncore * x0_value

    n_elements = shape[0] * shape[1]
    atomic_add_supply[shape[0], 1, 1](x0, x1, n_elements, BLOCK_SIZE=shape[1])
    test_common.validate_cmp(dtype, x1, x1_ref)

if __name__ == "__main__":
    param_list = ['float32', (32, 32), 2]
    test_atomic_add_2d(param_list)
