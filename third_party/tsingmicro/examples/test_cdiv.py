import pytest
import torch

import triton
import triton.language as tl
import benchmark

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def cdiv_kernel(
    x_ptr,
    y_ptr,
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

    # Compute the absolute value
    out = tl.cdiv(x, y)

    # Store the result
    tl.store(output_ptr + offsets, out, mask=mask)


def cdiv_triton(x, y):
    # Get the number of elements
    n_elements = x.numel()

    # Allocate output tensor
    output = torch.empty_like(x)

    # Define block size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    print("grid value is ", grid)

    x = x.to(DEVICE)
    y = y.to(DEVICE)
    output = output.to(DEVICE)
    # Launch the kernel
    cdiv_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=1024,
    )
    output = output.to('cpu')
    return output


@pytest.mark.parametrize("size, dtype", [  #
    (size, dtype) for size in [98432] for dtype in [torch.int32]
])
def test_cdiv(size, dtype, device="cpu"):
    # Generate random input tensor
    x = torch.randint(1, 100, (size, ), device=device, dtype=dtype)
    y = torch.randint(1, 100, (size, ), device=device, dtype=dtype)

    # Call the Triton kernel
    output = cdiv_triton(x, y)

    # Verify the result
    expected = (x + y - 1) // y

    # compare
    print(f"The maximum difference between torch and triton is "
          f"{torch.max(torch.abs(expected - output))}")
    # Verify the result
    torch.testing.assert_close(output, expected, atol=1e-2, rtol=0)


@benchmark.measure()
def benchmark_cdiv_triton(size, dtype, provider):
    if provider != "triton":
        raise ValueError("This benchmark is only for the Triton provider.")

    # Generate random input data
    x = torch.randint(1, 100, (size, ), device="cpu", dtype=dtype)
    y = torch.randint(1, 100, (size, ), device="cpu", dtype=dtype)

    # Run the Triton kernel
    output = cdiv_triton(x, y)

    # Verify the result
    torch.testing.assert_close(output, (x + y - 1) // y, atol=1e-2, rtol=0)


if __name__ == "__main__":
    # benchmark.select_cpu_backend()
    for X in [2**i for i in range(22, 25, 1)]:
        benchmark_cdiv_triton(X, torch.int32, "triton")
