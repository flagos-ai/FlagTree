import torch

import triton
import triton.language as tl

# The purpose of this kernel and test is to catch incorrectly optimized kernels
# where copy elimination happens erroneously in the absence of explicit memory allocation.
# Such optimization bugs can result in incorrect behavior when swapping two arrays,
# particularly when both arrays unintentionally end up with the same data due to
# missing intermediate storage or mismanaged memory access.

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


def swap(x: torch.Tensor, y: torch.Tensor):
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    swap_kernel[grid](x, y, BLOCK_SIZE=1024)


def test(device):
    torch.manual_seed(0)
    size = 10240
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    assert not torch.equal(x, y)
    x_ = x.clone()
    y_ = y.clone()
    swap(x, y)
    assert torch.equal(x, y_)
    assert torch.equal(y, x_)
