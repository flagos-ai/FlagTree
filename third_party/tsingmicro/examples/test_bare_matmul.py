# this is a benchmark which multiplies square matrices with maximum block size
# to check the performance of tl.dot operation

import pytest
import torch
import triton
import triton.language as tl
import benchmark

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def bare_matmul(X, Y, Z, M, N, K, BLOCK_SIZE: tl.constexpr):
    pid_x = tl.program_id(0)  # block row id
    pid_y = tl.program_id(1)  # block column id

    offs_x = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_y = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(X + offs_x[:, None] * K + offs_y[None, :])
    y = tl.load(Y + offs_x[:, None] * N + offs_y[None, :])

    z = tl.dot(x, y)

    tl.store(Z + offs_x[:, None] * N + offs_y[None, :], z)


@benchmark.measure()
def bench_matmul(N, provider):
    device = 'cpu'
    dtype = torch.float32
    a = torch.randint(0, 100, (N, N), device=device, dtype=dtype)
    b = torch.randint(0, 100, (N, N), device=device, dtype=dtype)
    c = torch.empty((N, N), device=device, dtype=dtype)
    if provider == 'torch' or provider == 'test':
        c_ref = torch.matmul(a, b)
    if provider == 'triton' or provider == 'test':
        bare_matmul[(1, )](a, b, c, N, N, N, N)
        if provider == 'test':
            torch.testing.assert_close(c, c_ref, atol=1e-2, rtol=0)


@pytest.mark.parametrize("N, dtype", [  #
    (N, dtype) for N in [64, 128, 256] for dtype in [torch.float32]
])
def test_bare_matmul(N, dtype, device='cpu'):
    a = torch.randint(0, 100, (N, N), device=device, dtype=dtype)
    b = torch.randint(0, 100, (N, N), device=device, dtype=dtype)
    c = torch.empty((N, N), device=device, dtype=dtype)
    bare_matmul[(1, )](a, b, c, N, N, N, N)
    c_ref = torch.matmul(a, b)

    # compare
    print(f"The maximum difference between torch and triton is "
          f"{torch.max(torch.abs(c_ref - c))}")
    torch.testing.assert_close(c, c_ref, atol=1e-5, rtol=0)


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    test_bare_matmul(128, torch.float32)
    # for X in [2**i for i in range(7, 10, 1)]:
    #     for provider in ['test', 'torch', 'triton']:
    #         bench_matmul(X, provider)
