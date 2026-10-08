# flagtree tle
"""
TLE Normalize Integration Tests

Tests TLE normalize functionality and row-wise normalization operations:
- Mean/variance normalization using TLE pipeline
- Two-pass reduction with shared memory staging
- Integration with Triton JIT and TLE operations
- Memory allocation and bidirectional copy validation for normalize workloads
"""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language.gpu as tle


@triton.jit
def normalize_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EPS: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    x_smem = tle.alloc([BLOCK_M, BLOCK_N], dtype=tl.float32, layout=None, scope=tle.smem, nv_mma_shared_layout=False)
    y_smem = tle.alloc([BLOCK_M, BLOCK_N], dtype=tl.float32, layout=None, scope=tle.smem, nv_mma_shared_layout=False)
    row_ids = tl.arange(0, BLOCK_M)[:, None]
    col_ids = tl.arange(0, BLOCK_N)[None, :]
    row_ids = tl.broadcast_to(row_ids, (BLOCK_M, BLOCK_N))
    col_ids = tl.broadcast_to(col_ids, (BLOCK_M, BLOCK_N))
    x_smem_ptrs = tle.local_ptr(x_smem, (row_ids, col_ids))
    y_smem_ptrs = tle.local_ptr(y_smem, (row_ids, col_ids))

    # Pass 1: accumulate sum and sum of squares over the row
    sum_acc = tl.zeros((BLOCK_M, ), dtype=tl.float32)
    sq_acc = tl.zeros((BLOCK_M, ), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn

        tle.copy(x_ptrs, x_smem, [BLOCK_M, BLOCK_N])
        x_tile = tl.load(x_smem_ptrs)
        sum_acc += tl.sum(x_tile, axis=1)
        sq_acc += tl.sum(x_tile * x_tile, axis=1)

    mean = sum_acc / N
    var = sq_acc / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + EPS)

    # Pass 2: normalize each tile and store it back
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn

        tle.copy(x_ptrs, x_smem, [BLOCK_M, BLOCK_N])
        x_tile = tl.load(x_smem_ptrs)
        y_tile = (x_tile - mean[:, None]) * rstd[:, None]
        tl.store(y_smem_ptrs, y_tile)
        tle.copy(y_smem, y_ptrs, [BLOCK_M, BLOCK_N])


def tle_normalize(X, Y, BLOCK_M=64, BLOCK_N=64, eps=1e-5):
    assert X.shape == Y.shape, "Input and output tensor shapes must match"

    M, N = X.shape

    stride_xm = X.stride(0)
    stride_xn = X.stride(1)
    stride_ym = Y.stride(0)
    stride_yn = Y.stride(1)

    grid = (triton.cdiv(M, BLOCK_M), )
    print(f"Launching normalize kernel with grid: {grid}")
    print(f"X stride: {stride_xm}, {stride_xn}")
    print(f"Y stride: {stride_ym}, {stride_yn}")

    normalize_kernel[grid](X, Y, M, N, stride_xm, stride_xn, stride_ym, stride_yn, BLOCK_M, BLOCK_N, eps)


class TestTLENormalize:
    """TLE Normalize Integration Tests"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
    def test_normalize_basic(self):
        """Test basic row-wise normalize functionality"""
        torch.manual_seed(42)  # Ensure reproducibility

        M, N = 512, 512
        BLOCK_M, BLOCK_N = 64, 64

        # Create test data
        x = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
        y = torch.empty(M, N, device="cuda", dtype=torch.float32).contiguous()

        # Execute TLE normalize computation
        tle_normalize(x, y, BLOCK_M, BLOCK_N)

        # Verify results
        eps = 1e-5
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        expected = (x - mean) / torch.sqrt(var + eps)
        torch.testing.assert_close(y, expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
