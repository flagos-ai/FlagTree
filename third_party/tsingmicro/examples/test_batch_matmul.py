import pytest
import torch

import triton
import triton.language as tl
import benchmark


@triton.jit
def batch_matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_ae, stride_am, stride_ak,  #
        stride_be, stride_bk, stride_bn,  #
        stride_ce, stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    pid_b = tl.program_id(1)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + pid_b * stride_ae + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + pid_b * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float32)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + pid_b * stride_ce + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def batch_matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[0] == b.shape[0], "Batch dim mismatch"
    assert a.shape[2] == b.shape[1], "K dim mismatch"
    batch, M, K = a.shape
    _, _, N = b.shape

    assert a.is_contiguous(), "Matrix A must be contiguous"
    # assert b.is_contiguous(), "Matrix B must be contiguous"

    # Allocates output.
    c = torch.empty((batch, M, N), dtype=a.dtype, device=a.device)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        batch,
    )
    batch_matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1), a.stride(2),  #
        b.stride(0), b.stride(1), b.stride(2),  #
        c.stride(0), c.stride(1), c.stride(2),  #
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=8)
    return c


@pytest.mark.parametrize("M, K, N, dtype", [  #
    (M, K, N, dtype)
    for M in [48, 64, 128]
    for K in [128, 156, 512]
    for N in [48, 64, 128]
    for dtype in [torch.float32, torch.float16, torch.bfloat16]
])
def test_batch_matmul(M, K, N, dtype, device='cpu'):
    # Generate random input tensors
    torch.manual_seed(0)
    rows1 = M
    cols1 = K
    rows2 = K
    cols2 = N
    batch = 4
    a = torch.randint(0, 100, (batch, rows1, cols1), device=device, dtype=dtype)
    b = torch.randint(0, 100, (batch, cols2, rows2), device=device, dtype=dtype).transpose(1, 2)

    print("\n ", b.stride())

    triton_output = batch_matmul(a, b)
    torch_output = torch.bmm(a, b)
    # compare
    print(f"The maximum difference between torch and triton is "
          f"{torch.max(torch.abs(torch_output - triton_output))}")
    torch.testing.assert_close(triton_output, torch_output, atol=1e-2, rtol=0)
