import pytest
import torch

import triton
import triton.language as tl
import benchmark



@triton.jit
def fma_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
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
    y = tl.load(y_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)

    # Compute the absolute value
    out = tl.fma(x, y, z)

    # Store the result
    tl.store(output_ptr + offsets, out, mask=mask)


def fma_triton(x, y, z):
    # Get the number of elements
    n_elements = x.numel()

    # Allocate output tensor
    output = torch.empty_like(x)

    # Define block size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    print("grid value is ", grid)

    # Launch the kernel
    fma_kernel[grid](
        x,
        y,
        z,
        output,
        n_elements,
        BLOCK_SIZE=1024,
    )

    return output

@pytest.mark.parametrize("size, dtype", [  #
    (size, dtype)
    for size in [1024]
    for dtype in [torch.float32]
])
def test_fma(size, dtype, device="cpu"):
    # Generate random input data
    x = torch.randn(size, device=device, dtype=dtype)
    y = torch.randn(size, device=device, dtype=dtype)
    z = torch.randn(size, device=device, dtype=dtype)

    # Call the Triton implementation
    output = fma_triton(x, y, z)

    # Verify the output
    expected = torch.mul(x, y) + z

    # compare
    print(
        f"The maximum difference between torch and triton is "
        f"{torch.max(torch.abs(expected - output))}"
    )
    torch.testing.assert_close(output, expected, atol=1e-2, rtol=0)



@benchmark.measure()
def benchmark_fma_triton(size, dtype, provider):
    if provider != "triton":
        raise ValueError("This benchmark is only for the Triton provider.")

    # Generate random input data
    x = torch.randn(size, device="cpu", dtype=dtype)
    y = torch.randn(size, device="cpu", dtype=dtype)
    z = torch.randn(size, device="cpu", dtype=dtype)

    # Call the Triton implementation
    output = fma_triton(x, y, z)

    # Verify the output
    expected = torch.mul(x, y) + z
    torch.testing.assert_close(output, expected, atol=1e-2, rtol=0)


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for size in [i**2 for i in range(22, 25, 1)]:
        benchmark_fma_triton(size, torch.float32, "triton")
