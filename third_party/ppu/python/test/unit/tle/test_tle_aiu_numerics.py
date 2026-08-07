"""Numerical correctness tests for tle.load(is_async=True) on PPU.

Two groups:
  A. Block-pointer path  → AIU hardware DMA
  B. Non-block-pointer path → cp.async fallback (TleLowerAsyncLoad)

All tests run on the actual PPU device and validate against torch references.
"""

import pytest
import torch
import triton
import triton.language as tl

try:
    import triton.experimental.tle.language as tle
    HAS_TLE = True
except ImportError:
    HAS_TLE = False

pytestmark = pytest.mark.skipif(not HAS_TLE, reason="TLE not available")


def _has_device():
    try:
        torch.zeros(1, device="cuda")
        return True
    except Exception:
        return False


_skip_no_device = pytest.mark.skipif(not _has_device(), reason="No PPU/CUDA device")

# ==========================================================================
# A. Block-pointer path (AIU)
# ==========================================================================

# --------------------------------------------------------------------------
# A1. Load identity: output == input (bit-exact)
# --------------------------------------------------------------------------


@triton.jit
def _bp_load_kernel(a_ptr, c_ptr, M, K, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_k = pid // num_pid_m
    a_bp = tl.make_block_ptr(a_ptr, shape=(M, K), strides=(K, 1), offsets=(pid_m * BLOCK_M, pid_k * BLOCK_K),
                             block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    a = tle.load(a_bp, is_async=True)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    c_ptrs = c_ptr + K * offs_m[:, None] + offs_k[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(c_ptrs, a, mask=c_mask)


@_skip_no_device
@pytest.mark.parametrize("BLOCK_M, BLOCK_K", [(32, 32), (64, 64), (128, 64), (64, 128), (128, 128)])
@pytest.mark.parametrize("num_warps", [2, 4])
@pytest.mark.parametrize("num_stages", [2, 4])
def test_bp_load_identity(BLOCK_M, BLOCK_K, num_warps, num_stages):
    M, K = 1024, 1024
    A = torch.randn((M, K), dtype=torch.float16, device="cuda")
    C = torch.empty_like(A)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(K, BLOCK_K), )
    _bp_load_kernel[grid](A, C, M, K, BLOCK_M, BLOCK_K, num_warps=num_warps, num_stages=num_stages)
    torch.testing.assert_close(A, C, rtol=0, atol=0)


# --------------------------------------------------------------------------
# A3. Multiple data types
# --------------------------------------------------------------------------


@_skip_no_device
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bp_load_dtypes(dtype):
    M, K = 512, 512
    A = torch.randn((M, K), dtype=dtype, device="cuda")
    C = torch.empty_like(A)
    grid = (triton.cdiv(M, 64) * triton.cdiv(K, 64), )
    _bp_load_kernel[grid](A, C, M, K, 64, 64, num_warps=4, num_stages=2)
    torch.testing.assert_close(A, C, rtol=0, atol=0)


# --------------------------------------------------------------------------
# A4. Element-wise arithmetic: two AIU loads + add
# --------------------------------------------------------------------------


@triton.jit
def _bp_add_kernel(a_ptr, b_ptr, c_ptr, M, K, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_k = pid // num_pid_m
    a_bp = tl.make_block_ptr(a_ptr, shape=(M, K), strides=(K, 1), offsets=(pid_m * BLOCK_M, pid_k * BLOCK_K),
                             block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_bp = tl.make_block_ptr(b_ptr, shape=(M, K), strides=(K, 1), offsets=(pid_m * BLOCK_M, pid_k * BLOCK_K),
                             block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    a = tle.load(a_bp, is_async=True)
    b = tle.load(b_bp, is_async=True)
    c = a + b
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    c_ptrs = c_ptr + K * offs_m[:, None] + offs_k[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(c_ptrs, c, mask=c_mask)


@_skip_no_device
@pytest.mark.parametrize("BLOCK_M, BLOCK_K", [(32, 32), (64, 64), (128, 128)])
def test_bp_elementwise_add(BLOCK_M, BLOCK_K):
    M, K = 1024, 1024
    A = torch.randn((M, K), dtype=torch.float16, device="cuda")
    B = torch.randn((M, K), dtype=torch.float16, device="cuda")
    C = torch.empty_like(A)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(K, BLOCK_K), )
    _bp_add_kernel[grid](A, B, C, M, K, BLOCK_M, BLOCK_K, num_warps=4, num_stages=2)
    torch.testing.assert_close(C, A + B, rtol=0, atol=0)


# --------------------------------------------------------------------------
# A4b. Load with order=(0, 1): data appears transposed vs row-major storage
# --------------------------------------------------------------------------


@triton.jit
def _bp_load_colmajor_kernel(a_ptr, c_ptr, M, K, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """Load from a column-major tensor using order=(0,1) — matching strides."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_k = pid // num_pid_m
    # strides=(1, M) matches order=(0,1): dim 0 is contiguous
    a_bp = tl.make_block_ptr(a_ptr, shape=(M, K), strides=(1, M), offsets=(pid_m * BLOCK_M, pid_k * BLOCK_K),
                             block_shape=(BLOCK_M, BLOCK_K), order=(0, 1))
    a = tle.load(a_bp, is_async=True)
    # Store back using standard row-major pointers
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    c_ptrs = c_ptr + K * offs_m[:, None] + offs_k[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(c_ptrs, a, mask=c_mask)


@_skip_no_device
@pytest.mark.parametrize("BLOCK", [64, 128])
def test_bp_load_colmajor_identity(BLOCK):
    """order=(0,1) with column-major strides: loaded data matches original."""
    M, K = 256, 256
    # Create column-major tensor (Fortran order)
    A_colmajor = torch.randn((M, K), dtype=torch.float16, device="cuda").t().contiguous().t()
    C = torch.empty((M, K), dtype=torch.float16, device="cuda")
    grid = (triton.cdiv(M, BLOCK) * triton.cdiv(K, BLOCK), )
    _bp_load_colmajor_kernel[grid](A_colmajor, C, M, K, BLOCK, BLOCK, num_warps=4, num_stages=2)
    torch.testing.assert_close(C, A_colmajor, rtol=0, atol=0)


# --------------------------------------------------------------------------
# A4c. GEMM with order=(0, 1) on both operands
# --------------------------------------------------------------------------


@triton.jit
def _bp_gemm_order01_kernel(
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
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    a_bp = tl.make_block_ptr(a_ptr, shape=(M, K), strides=(stride_am, stride_ak), offsets=(pid_m * BLOCK_M, 0),
                             block_shape=(BLOCK_M, BLOCK_K), order=(0, 1))
    b_bp = tl.make_block_ptr(b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), offsets=(0, pid_n * BLOCK_N),
                             block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tle.load(a_bp, is_async=True)
        b = tle.load(b_bp, is_async=True)
        acc = tl.dot(a, b, acc=acc)
        a_bp = tl.advance(a_bp, (0, BLOCK_K))
        b_bp = tl.advance(b_bp, (BLOCK_K, 0))
    c = acc.to(tl.float16)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@_skip_no_device
@pytest.mark.parametrize("BLOCK", [32, 64])
def test_bp_gemm_order01(BLOCK):
    """GEMM with order=(0,1) on square matrices: result = A.t() @ B.t()."""
    S = 512
    torch.manual_seed(42)
    A = torch.randn((S, S), dtype=torch.float16, device="cuda")
    B = torch.randn((S, S), dtype=torch.float16, device="cuda")
    C = torch.empty((S, S), dtype=torch.float16, device="cuda")
    grid = (triton.cdiv(S, BLOCK) * triton.cdiv(S, BLOCK), )
    _bp_gemm_order01_kernel[grid](A, B, C, S, S, S, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0),
                                  C.stride(1), BLOCK, BLOCK, BLOCK, num_warps=4, num_stages=2)
    ref = torch.matmul(A.t().float(), B.t().float()).half()
    torch.testing.assert_close(C, ref, rtol=1e-2, atol=1e-2)


# --------------------------------------------------------------------------
# A5. GEMM: both operands via AIU block-pointer
# --------------------------------------------------------------------------


@triton.jit
def _bp_gemm_kernel(
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
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    a_bp = tl.make_block_ptr(a_ptr, shape=(M, K), strides=(stride_am, stride_ak), offsets=(pid_m * BLOCK_M, 0),
                             block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_bp = tl.make_block_ptr(b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), offsets=(0, pid_n * BLOCK_N),
                             block_shape=(BLOCK_K, BLOCK_N), order=(1, 0))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tle.load(a_bp, is_async=True)
        b = tle.load(b_bp, is_async=True)
        acc = tl.dot(a, b, acc=acc)
        a_bp = tl.advance(a_bp, (0, BLOCK_K))
        b_bp = tl.advance(b_bp, (BLOCK_K, 0))
    c = acc.to(tl.float16)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def _run_bp_gemm(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps=4, num_stages=2):
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device="cuda")
    B = torch.randn((K, N), dtype=torch.float16, device="cuda")
    C = torch.empty((M, N), dtype=torch.float16, device="cuda")
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    _bp_gemm_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0),
                          C.stride(1), BLOCK_M, BLOCK_N, BLOCK_K, num_warps=num_warps, num_stages=num_stages)
    ref = torch.matmul(A.float(), B.float()).half()
    return C, ref


@_skip_no_device
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(32, 32, 32), (64, 64, 64), (128, 64, 64), (128, 128, 64)])
@pytest.mark.parametrize("num_warps", [2, 4])
def test_bp_gemm(BLOCK_M, BLOCK_N, BLOCK_K, num_warps):
    C, ref = _run_bp_gemm(1024, 1024, 1024, BLOCK_M, BLOCK_N, BLOCK_K, num_warps=num_warps)
    torch.testing.assert_close(C, ref, rtol=1e-2, atol=1e-2)


@_skip_no_device
@pytest.mark.parametrize("num_stages", [1, 2, 4])
def test_bp_gemm_pipeline_stages(num_stages):
    C, ref = _run_bp_gemm(512, 512, 256, 64, 64, 64, num_stages=num_stages)
    torch.testing.assert_close(C, ref, rtol=1e-2, atol=1e-2)


# --------------------------------------------------------------------------
# A6. GEMM bf16
# --------------------------------------------------------------------------


@triton.jit
def _bp_gemm_bf16_kernel(
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
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    a_bp = tl.make_block_ptr(a_ptr, shape=(M, K), strides=(stride_am, stride_ak), offsets=(pid_m * BLOCK_M, 0),
                             block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_bp = tl.make_block_ptr(b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), offsets=(0, pid_n * BLOCK_N),
                             block_shape=(BLOCK_K, BLOCK_N), order=(1, 0))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tle.load(a_bp, is_async=True)
        b = tle.load(b_bp, is_async=True)
        acc = tl.dot(a, b, acc=acc)
        a_bp = tl.advance(a_bp, (0, BLOCK_K))
        b_bp = tl.advance(b_bp, (BLOCK_K, 0))
    c = acc.to(tl.bfloat16)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@_skip_no_device
def test_bp_gemm_bf16():
    M, N, K = 512, 512, 256
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    B = torch.randn((K, N), dtype=torch.bfloat16, device="cuda")
    C = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64), )
    _bp_gemm_bf16_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0),
                               C.stride(1), 64, 64, 64, num_warps=4, num_stages=2)
    ref = torch.matmul(A.float(), B.float()).bfloat16()
    torch.testing.assert_close(C, ref, rtol=1e-2, atol=1e-2)


# --------------------------------------------------------------------------
# A7. GEMM: mixed — A via AIU block-pointer, B via regular pointer
# --------------------------------------------------------------------------


@triton.jit
def _bp_gemm_mixed_kernel(
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
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    a_bp = tl.make_block_ptr(a_ptr, shape=(M, K), strides=(stride_am, stride_ak), offsets=(pid_m * BLOCK_M, 0),
                             block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        a = tle.load(a_bp, is_async=True)
        offs_bk = k_start + tl.arange(0, BLOCK_K)
        b_ptrs = b_ptr + stride_bk * offs_bk[:, None] + stride_bn * offs_bn[None, :]
        b_mask = (offs_bk[:, None] < K) & (offs_bn[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc=acc)
        a_bp = tl.advance(a_bp, (0, BLOCK_K))
    c = acc.to(tl.float16)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@_skip_no_device
@pytest.mark.parametrize("num_warps", [2, 4])
def test_bp_gemm_mixed_load(num_warps):
    M, N, K = 512, 512, 256
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device="cuda")
    B = torch.randn((K, N), dtype=torch.float16, device="cuda")
    C = torch.empty((M, N), dtype=torch.float16, device="cuda")
    grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64), )
    _bp_gemm_mixed_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0),
                                C.stride(1), 64, 64, 64, num_warps=num_warps, num_stages=2)
    ref = torch.matmul(A.float(), B.float()).half()
    torch.testing.assert_close(C, ref, rtol=1e-2, atol=1e-2)


# ==========================================================================
# B. Non-block-pointer path (cp.async fallback)
# ==========================================================================

# --------------------------------------------------------------------------
# B1. Load identity: non-block-pointer async load == input (bit-exact)
# --------------------------------------------------------------------------


@triton.jit
def _nbp_load_kernel(a_ptr, c_ptr, M, K, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_k = pid // num_pid_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + K * offs_m[:, None] + offs_k[None, :]
    a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    a = tle.load(a_ptrs, mask=a_mask, is_async=True)
    c_ptrs = c_ptr + K * offs_m[:, None] + offs_k[None, :]
    tl.store(c_ptrs, a, mask=a_mask)


@_skip_no_device
@pytest.mark.parametrize("BLOCK_M, BLOCK_K", [(32, 32), (64, 64), (128, 64)])
@pytest.mark.parametrize("num_warps", [2, 4])
@pytest.mark.parametrize("num_stages", [2, 4])
def test_nbp_load_identity(BLOCK_M, BLOCK_K, num_warps, num_stages):
    M, K = 1024, 1024
    A = torch.randn((M, K), dtype=torch.float16, device="cuda")
    C = torch.empty_like(A)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(K, BLOCK_K), )
    _nbp_load_kernel[grid](A, C, M, K, BLOCK_M, BLOCK_K, num_warps=num_warps, num_stages=num_stages)
    torch.testing.assert_close(A, C, rtol=0, atol=0)


# --------------------------------------------------------------------------
# B2. Multiple data types
# --------------------------------------------------------------------------


@_skip_no_device
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_nbp_load_dtypes(dtype):
    M, K = 512, 512
    A = torch.randn((M, K), dtype=dtype, device="cuda")
    C = torch.empty_like(A)
    grid = (triton.cdiv(M, 64) * triton.cdiv(K, 64), )
    _nbp_load_kernel[grid](A, C, M, K, 64, 64, num_warps=4, num_stages=2)
    torch.testing.assert_close(A, C, rtol=0, atol=0)


# --------------------------------------------------------------------------
# B3. Element-wise: two non-block-pointer async loads + add
# --------------------------------------------------------------------------


@triton.jit
def _nbp_add_kernel(a_ptr, b_ptr, c_ptr, M, K, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_k = pid // num_pid_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    ptrs = a_ptr + K * offs_m[:, None] + offs_k[None, :]
    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    a = tle.load(ptrs, mask=mask, is_async=True)
    b_ptrs = b_ptr + K * offs_m[:, None] + offs_k[None, :]
    b = tle.load(b_ptrs, mask=mask, is_async=True)
    c = a + b
    c_ptrs = c_ptr + K * offs_m[:, None] + offs_k[None, :]
    tl.store(c_ptrs, c, mask=mask)


@_skip_no_device
@pytest.mark.parametrize("BLOCK_M, BLOCK_K", [(32, 32), (64, 64), (128, 128)])
def test_nbp_elementwise_add(BLOCK_M, BLOCK_K):
    M, K = 1024, 1024
    A = torch.randn((M, K), dtype=torch.float16, device="cuda")
    B = torch.randn((M, K), dtype=torch.float16, device="cuda")
    C = torch.empty_like(A)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(K, BLOCK_K), )
    _nbp_add_kernel[grid](A, B, C, M, K, BLOCK_M, BLOCK_K, num_warps=4, num_stages=2)
    torch.testing.assert_close(C, A + B, rtol=0, atol=0)


# --------------------------------------------------------------------------
# B4. GEMM: both operands via non-block-pointer async load
# --------------------------------------------------------------------------


@triton.jit
def _nbp_gemm_kernel(
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
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + stride_am * offs_am[:, None] + stride_ak * offs_k[None, :]
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
        a = tle.load(a_ptrs, mask=a_mask, other=0.0, is_async=True)
        b_ptrs = b_ptr + stride_bk * offs_k[:, None] + stride_bn * offs_bn[None, :]
        b_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)
        b = tle.load(b_ptrs, mask=b_mask, other=0.0, is_async=True)
        acc = tl.dot(a, b, acc=acc)
    c = acc.to(tl.float16)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@_skip_no_device
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(32, 32, 32), (64, 64, 64), (128, 64, 64)])
def test_nbp_gemm(BLOCK_M, BLOCK_N, BLOCK_K):
    M, N, K = 512, 512, 256
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device="cuda")
    B = torch.randn((K, N), dtype=torch.float16, device="cuda")
    C = torch.empty((M, N), dtype=torch.float16, device="cuda")
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    _nbp_gemm_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0),
                           C.stride(1), BLOCK_M, BLOCK_N, BLOCK_K, num_warps=4, num_stages=2)
    ref = torch.matmul(A.float(), B.float()).half()
    torch.testing.assert_close(C, ref, rtol=1e-2, atol=1e-2)


@_skip_no_device
@pytest.mark.parametrize("num_stages", [1, 2, 4])
def test_nbp_gemm_pipeline_stages(num_stages):
    M, N, K = 512, 512, 256
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device="cuda")
    B = torch.randn((K, N), dtype=torch.float16, device="cuda")
    C = torch.empty((M, N), dtype=torch.float16, device="cuda")
    grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64), )
    _nbp_gemm_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0),
                           C.stride(1), 64, 64, 64, num_warps=4, num_stages=num_stages)
    ref = torch.matmul(A.float(), B.float()).half()
    torch.testing.assert_close(C, ref, rtol=1e-2, atol=1e-2)


# --------------------------------------------------------------------------
# B5. GEMM mixed: A via non-block-pointer async, B via regular load
# --------------------------------------------------------------------------


@triton.jit
def _nbp_gemm_mixed_kernel(
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
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + stride_am * offs_am[:, None] + stride_ak * offs_k[None, :]
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
        a = tle.load(a_ptrs, mask=a_mask, other=0.0, is_async=True)
        b_ptrs = b_ptr + stride_bk * offs_k[:, None] + stride_bn * offs_bn[None, :]
        b_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc=acc)
    c = acc.to(tl.float16)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@_skip_no_device
@pytest.mark.parametrize("num_warps", [2, 4])
def test_nbp_gemm_mixed_load(num_warps):
    M, N, K = 512, 512, 256
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device="cuda")
    B = torch.randn((K, N), dtype=torch.float16, device="cuda")
    C = torch.empty((M, N), dtype=torch.float16, device="cuda")
    grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64), )
    _nbp_gemm_mixed_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0),
                                 C.stride(1), 64, 64, 64, num_warps=num_warps, num_stages=2)
    ref = torch.matmul(A.float(), B.float()).half()
    torch.testing.assert_close(C, ref, rtol=1e-2, atol=1e-2)


# ==========================================================================
# C. Cross-path: block-pointer AIU vs non-block-pointer produce same result
# ==========================================================================


@_skip_no_device
def test_bp_vs_nbp_gemm_same_result():
    """Both paths should produce numerically identical GEMM results."""
    M, N, K = 512, 512, 256
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device="cuda")
    B = torch.randn((K, N), dtype=torch.float16, device="cuda")
    C_bp = torch.empty((M, N), dtype=torch.float16, device="cuda")
    C_nbp = torch.empty((M, N), dtype=torch.float16, device="cuda")

    grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64), )
    _bp_gemm_kernel[grid](A, B, C_bp, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C_bp.stride(0),
                          C_bp.stride(1), 64, 64, 64, num_warps=4, num_stages=2)
    _nbp_gemm_kernel[grid](A, B, C_nbp, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C_nbp.stride(0),
                           C_nbp.stride(1), 64, 64, 64, num_warps=4, num_stages=2)
    torch.testing.assert_close(C_bp, C_nbp, rtol=0, atol=0)
