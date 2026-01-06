import pytest
import torch

import triton
import triton.language as tl
import benchmark


@triton.jit
def sqrt_rn_kernel(
    x_ptr,
    output_ptr,
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
    out = tl.sqrt_rn(x)

    # Store the result
    tl.store(output_ptr + offsets, out, mask=mask)


def sqrt_rn_triton(x):
    # Get the number of elements
    n_elements = x.numel()

    # Allocate output tensor
    output = torch.empty_like(x)

    # Define block size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    print("grid value is ", grid)

    # Launch the kernel
    sqrt_rn_kernel[grid](
        x,
        output,
        n_elements,
        BLOCK_SIZE=1024,
    )

    return output

@pytest.mark.parametrize("size, dtype", [  #
    (size, dtype)
    for size in [98432]
    for dtype in [torch.float32]
])
def test_sqrt_rn(size, dtype, device="cpu"):
    # Generate random input data
    x = torch.randn(size, device=device, dtype=dtype)

    # Call the Triton kernel
    output = sqrt_rn_triton(x)

    # Verify the output
    expected = torch.sqrt(x)
    torch.testing.assert_close(output, expected, equal_nan=True, atol=1e-2, rtol=0)

@benchmark.measure()
def benchmark_sqrt_rn_triton(size, dtype, provider):
    if provider != "triton":
        raise ValueError("This benchmark is only for the Triton provider.")

    # Generate random input data
    x = torch.randn(size, device="cpu", dtype=dtype)

    # Call the Triton kernel
    output = sqrt_rn_triton(x)

    # Verify the output
    # TODO：rounding_mode need double check
    expected = torch.sqrt(x)
    torch.testing.assert_close(output, expected, equal_nan=True, atol=1e-2, rtol=0)


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for size in [i**2 for i in range(22, 25, 1)]:
        benchmark_sqrt_rn_triton(size, torch.float32, "triton")
