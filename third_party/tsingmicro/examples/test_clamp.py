import pytest
import torch

import triton
import triton.language as tl
import benchmark as benchmark


@triton.jit
def clamp_kernel(
    x_ptr,
    output_ptr,
    min_val,
    max_val,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)

    # Calculate the start and offsets
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to avoid out-of-bounds access
    mask = offsets < n_elements

    # Load the input data
    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute the absolute value
    out = tl.clamp(x, min_val, max_val)

    # Store the result
    tl.store(output_ptr + offsets, out, mask=mask)


def clamp_triton(x, min_val, max_val):
    # Get the number of elements
    n_elements = x.numel()

    # Allocate output tensor
    output = torch.empty_like(x)

    # Define block size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    print("grid value is ", grid)

    # Launch the kernel
    clamp_kernel[grid](
        x,
        output,
        min_val,
        max_val,
        n_elements,
        BLOCK_SIZE=1024,
    )

    return output


@pytest.mark.parametrize("size, dtype", [  #
    (size, dtype) for size in [98432] for dtype in [torch.float32]
])
def test_clamp(size, dtype, device="cpu"):
    # Generate random input data
    x = torch.randn(size, device=device, dtype=dtype)
    min_val = -1.0
    max_val = 1.0

    # Call the Triton kernel
    output = clamp_triton(x, min_val, max_val)

    # Verify the output
    expected = torch.clamp(x, min_val, max_val)
    torch.testing.assert_close(output, expected, atol=1e-2, rtol=0)


@benchmark.measure()
def benchmark_clamp_triton(size, dtype, provider):
    if provider != "triton":
        raise ValueError("This benchmark is only for the Triton provider.")

    # Generate random input data
    x = torch.randn(size, device="cpu", dtype=dtype)
    min_val = -1.0
    max_val = 1.0

    # Call the Triton kernel
    output = clamp_triton(x, min_val, max_val)

    # Verify the output
    expected = torch.clamp(x, min_val, max_val)
    torch.testing.assert_close(output, expected, atol=1e-2, rtol=0)

    return output


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for size in [2**i for i in range(22, 25, 1)]:
        benchmark_clamp_triton(size, torch.float32, "triton")
