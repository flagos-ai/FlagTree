import torch
import triton
import triton.language as tl
import torch_npu
import pytest


@triton.jit
def swap_kernel(
    x_ptr,  # *Pointer* to first inout vector.
    y_ptr,  # *Pointer* to second inout vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    # NOTE: `constexpr` so it can be used as a shape value.
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    tl.store(x_ptr + offsets, y)
    tl.store(y_ptr + offsets, x)



def swap(x: torch.Tensor, y: torch.Tensor, size):
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    swap_kernel[grid](x, y, BLOCK_SIZE=size)


@pytest.mark.parametrize('shape', [(1,), (3,), (4,), (7,), (8,), (11,), (16,), (512,), (1024,)])
def test(shape):
    x = torch.rand(shape).npu()
    y = torch.rand(shape).npu()
    assert not torch.equal(x, y)
    x_ = x.clone()
    y_ = y.clone()
    swap(x, y, shape[0])
    torch.testing.assert_close(x, y_, rtol=1e-04, atol=1e-04, equal_nan=True)
    torch.testing.assert_close(y, x_, rtol=1e-04, atol=1e-04, equal_nan=True)