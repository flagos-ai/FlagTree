import triton
import triton.language as tl
import pytest
import test_common
import torch
import torch_npu


@triton.jit
def atomic_xchg(in_ptr0, out_ptr0, out_ptr1, n_elements, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.atomic_xchg(out_ptr0 + (x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp1, xmask)


@pytest.mark.parametrize('param_list',
                         [
                             ['int32', (32, 32), 2],
                             ['int16', (16, 16), 2],
                             ['int8', (32, 32), 2],
                             ['float32', (16, 16), 2],
                             ['float16', (64, 64), 4],
                             ['float32', (128, 128), 8],
                             ['float16', (128, 128), 16],
                             ['float32', (32768, 16), 32],
                         ]
                         )
def test_atomic_xchg(param_list):
    dtype, shape, ncore = param_list
    block_size = shape[0] * shape[1] // ncore
    split_size = shape[0] // ncore

    val = torch.randint(low=0, high=10, size=shape, dtype=eval(f'torch.{dtype}')).npu()

    pointer = torch.randint(low=0, high=10, size=(split_size, shape[1]), dtype=eval(f'torch.{dtype}')).npu()
    pointer_ref = pointer.clone()
    pointer_old = torch.full_like(val, -10).npu()
    pointer_old_ref = pointer_old.clone()

    pointer_ref = val[((ncore - 1) * split_size):(ncore * split_size)].clone()
    pointer_old_ref[0:split_size] = pointer
    pointer_old_ref[split_size:((ncore - 1) * split_size)] = val[0:(ncore - 2) * split_size]
    
    n_elements = shape[0] * shape[1]
    atomic_xchg[ncore, 1, 1](val, pointer, pointer_old, n_elements, BLOCK_SIZE=split_size * shape[1])
    test_common.validate_cmp(dtype, pointer, pointer_ref)
