"""
Vector Add
=============
"""

import os

import torch
import torch_npu
import triton
import triton.language as tl
from triton.testing import do_bench_npu


@triton.autotune(
    configs=[],
    key={"x": "n_elements"},
    split_params={"x": "BLOCK_SIZE"},
    tiling_params={},
    low_dims=["x"],
    persistent_reduction=False,
    dual_reduction=False,
)
@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    # NOTE: `constexpr` so it can be used as a shape value.
):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


def add_torch(x, y):
    return x + y


def add_autotune(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements)
    return output


def test_add(size: int):
    os.environ["TRITON_BENCH_METHOD"] = (
        "npu"  # use torch_npu.profiler to get calculating time
    )
    x = torch.rand(size, device="npu")
    y = torch.rand(size, device="npu")

    output_torch = add_torch(x, y)
    output_triton = add_autotune(x, y)
    assert torch.allclose(output_triton, output_torch)

    time_eager = do_bench_npu(lambda: add_torch(x, y))
    time_triton = do_bench_npu(lambda: add_autotune(x, y))
    assert (time_eager / time_triton) >= 0.8
    print(f"Vector Add {size} PASSED!")


if __name__ == "__main__":
    test_add(98432)
