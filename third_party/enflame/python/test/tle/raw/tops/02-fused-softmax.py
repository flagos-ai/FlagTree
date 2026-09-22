"""
TLE-Raw TOPS Backend: Fused Softmax
====================================

This tutorial demonstrates how to use TLE-Raw with the TOPS C++ backend
to compile and run a fused softmax kernel on GCU hardware.

The .tops file contains a softmax implementation using TOPS C++ device code,
compiled via topscc to LLVM IR.
"""

from pathlib import Path
import torch
import triton
import triton.language as tl
from triton.experimental.tle.raw import dialect
import triton.experimental.tle.language.raw as tle_raw

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def softmax_kernel_python(output_ptr, input_ptr, n_rows, n_cols, input_row_stride, output_row_stride,
                          BLOCK_SIZE: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_start * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_start * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


@dialect(name="tops", file=Path(__file__).parent / "02-fused-softmax.tops", extern_func_name="SoftmaxKernel",
         deferred=True)
def edsl(*args, **kwargs):
    ...


def naive_softmax(x):
    x_max, _ = x.max(dim=1)
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    ret = numerator / denominator[:, None]
    return ret


@triton.jit
def softmax_kernel(output_ptr, input_ptr, n_rows, n_cols, input_row_stride, output_row_stride,
                   BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    row_start_ptr = input_ptr + row_start * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    tle_raw.call(edsl, [output_ptr, row, n_rows, n_cols, input_row_stride, output_row_stride], output_indices=[0])


def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    softmax_kernel[(n_rows, 1, 1)](y, x, n_rows, n_cols, x.stride(0), y.stride(0), BLOCK_SIZE, num_warps=1)
    return y


def python_softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    softmax_kernel_python[(n_rows, 1, 1)](y, x, n_rows, n_cols, x.stride(0), y.stride(0), BLOCK_SIZE, num_warps=1)
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(24, 256 * 16 + 32, device=DEVICE)
    y_triton = softmax(x)
    y_torch = python_softmax(x)
    assert torch.allclose(y_triton, y_torch, atol=1e-4, rtol=1e-4), (y_triton, y_torch)
    print("TOPS Fused Softmax: PASSED")
