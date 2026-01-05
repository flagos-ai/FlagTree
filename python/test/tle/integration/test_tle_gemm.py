# Copyright (c) 2025  XCoreSigma Inc. All rights reserved.
# flagtree tle
"""
TLE GEMM Integration Tests

Tests TLE GEMM functionality and matrix multiplication operations:
- GEMM operations using TLE pipeline
- Matrix multiplication with shared memory optimization
- Integration with Triton JIT and TLE operations
- Memory allocation and computation validation for GEMM workloads
"""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle
# Disable TF32, force pure FP32 accumulation
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@triton.jit
def gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_smem = tle.alloc([BLOCK_M, BLOCK_N], dtype=tl.float32, layout=None, scope=tle.smem, nv_mma_shared_layout=False)
    b_smem = tle.alloc([BLOCK_M, BLOCK_N], dtype=tl.float32, layout=None, scope=tle.smem, nv_mma_shared_layout=False)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn

        tle.copy(a_ptrs, a_smem, [BLOCK_M, BLOCK_N])
        tle.copy(b_ptrs, b_smem, [BLOCK_M, BLOCK_N])
        a_tile = tle.local_load(a_smem)
        b_tile = tle.local_load(b_smem)
        accumulator += tl.dot(a_tile, b_tile, input_precision="ieee")

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def tle_gemm(A, B, C, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64):
    assert A.shape[1] == B.shape[0]
    assert C.shape == (A.shape[0], B.shape[1])

    M, K = A.shape
    K_B, N = B.shape
    assert K == K_B

    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    print(f"Launching GEMM kernel with grid: {grid}")
    print(f"A stride: {stride_am}, {stride_ak}")
    print(f"B stride: {stride_bk}, {stride_bn}")
    print(f"C stride: {stride_cm}, {stride_cn}")

    gemm_kernel[grid](A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M,
                      BLOCK_N, BLOCK_K)


class TestTLEGEMM:
    """TLE GEMM Integration Tests"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
    def test_gemm_basic(self):
        """Test basic GEMM functionality with square matrices"""
        torch.manual_seed(42)  # Ensure reproducibility

        M, N, K = 1024, 1024, 1024
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64

        # Create test matrices
        a = torch.randn(M, K, device="cuda", dtype=torch.float32).contiguous()
        b = torch.randn(K, N, device="cuda", dtype=torch.float32).contiguous()
        c = torch.empty(M, N, device="cuda", dtype=torch.float32).contiguous()

        # Execute TLE GEMM computation
        tle_gemm(a, b, c, BLOCK_M, BLOCK_N, BLOCK_K)

        # Verify results
        expected = torch.matmul(a, b)
        torch.testing.assert_close(c, expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
