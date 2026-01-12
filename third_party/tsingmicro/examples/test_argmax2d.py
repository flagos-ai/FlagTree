import torch
import triton
import triton.language as tl
import pytest
import benchmark

benchmark.select_cpu_backend()


@triton.jit
def argmax_kernel_2d(
    x_ptr,
    output_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get current block's row index
    pid_x = tl.program_id(0)

    # Calculate data offsets for current block
    offs_x = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_y = tl.arange(0, N)  # Iterate through all columns

    # Load input data
    x = tl.load(x_ptr + offs_x[:, None] * N + offs_y[None, :])

    # Calculate argmax for each row
    max_idx = tl.argmax(x, axis=1)

    # Convert result to int32 and store
    result = max_idx.to(tl.int32)
    tl.store(output_ptr + offs_x, result)


@pytest.mark.parametrize("N", [16, 32, 64])
def test_argmax(N, device):
    # Set input size
    x = torch.rand([N, N], device=device, dtype=torch.float32)
    output = torch.empty([N], device=device, dtype=torch.int32)

    # Run kernel
    argmax_kernel_2d[(1, )](x, output, N, BLOCK_SIZE=N)

    # Calculate reference result and verify
    ans = torch.argmax(x, dim=1).to(torch.int32)
    torch.testing.assert_close(output, ans, rtol=0, atol=0)


if __name__ == "__main__":
    test_argmax(16, "cpu")
