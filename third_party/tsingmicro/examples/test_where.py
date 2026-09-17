import torch

import triton
import triton.language as tl
import benchmark
import numpy as np


@triton.jit
def where_kernel(a_ptr, b_ptr, output_ptr, n_elements,
                 BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                 # NOTE: `constexpr` so it can be used as a shape value.
                 ):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    decide = a > (b + 0)
    output = tl.where(decide, a, b)
    tl.store(output_ptr + offsets, output, mask=mask)


def where(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    # assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    where_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output


def test_where_1(device):
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=device, dtype=torch.float32)
    y = 1 - x
    # y = torch.rand(size, device=device, dtype=torch.float32)
    cond = x > y
    output_torch = torch.where(cond, x, y)

    output_triton = where(x, y)

    max = torch.maximum(x, y)

    torch.testing.assert_close(output_torch, output_triton, atol=1e-5, rtol=0)

    torch.testing.assert_close(output_torch, max, atol=1e-5, rtol=0)

    # torch.testing.assert_close(output_triton, x, atol=1e-5, rtol=0)
