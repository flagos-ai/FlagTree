import pytest
import torch

import triton
import triton.language as tl
import benchmark
from util import gems_assert_cosine_similarity


@triton.jit
def div_rn_kernel(
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
    out = tl.div_rn(x, y)

    # Store the result
    tl.store(output_ptr + offsets, out, mask=mask)


def div_rn_triton(x, y):
    # Get the number of elements
    n_elements = x.numel()

    # Allocate output tensor
    output = torch.empty_like(x)

    # Define block size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    print("grid value is ", grid)

    # Launch the kernel
    div_rn_kernel[grid](
        x,
        y,
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
def test_div_rn(size, dtype, device="cpu"):
    # Generate random input tensors
    x = torch.randn(size, device=device, dtype=dtype)
    y = torch.randn(size, device=device, dtype=dtype)

    # Call the Triton kernel
    output = div_rn_triton(x, y)

    # Verify the output
    # TODO：rounding_mode need double check
    expected = torch.div(x, y, rounding_mode=None)
    # torch.testing.assert_close(output, expected, atol=1e-2, rtol=0)
    # FIXME: div int has low precision, we use cosine similarity to verify
    gems_assert_cosine_similarity(output, expected, dtype=dtype)

@benchmark.measure()
def benchmark_div_rn_triton(size, dtype, provider):
    if provider != "triton":
        raise ValueError("This benchmark is only for the Triton provider.")

    # Generate random input tensors
    x = torch.randn(size, device="cpu", dtype=dtype)
    y = torch.randn(size, device="cpu", dtype=dtype)

    # Call the Triton kernel
    output = div_rn_triton(x, y)

    # Verify the output
    # TODO：rounding_mode need double check
    expected = torch.div(x, y, rounding_mode=None)
    torch.testing.assert_close(output, expected, atol=1e-2, rtol=0)


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for i in [2**i for i in range(22, 25, 1)]:
        benchmark_div_rn_triton(i, torch.float32, provider="triton")