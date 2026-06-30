# flagtree tle
"""
TLE GEMM via local_ptr staging + tl.dot (sunrise)
=================================================

Exercises the most common TLE pattern that was previously untested on sunrise:
shared-memory staging through tle.gpu.local_ptr feeding tl.dot (a GEMM tile).

This also indirectly covers:
  * the local_ptr TTGIR pass chain (early_assign_memory_space / select_encodings
    / insert_local_pointer_barriers / optimize_local_pointer_loads/_stores),
  * the gpu.barrier -> STVM::Barrier0Op lowering between local store and load,
  * the AABS dot-dimension floor for sunrise (M>=8, N>=8, K>=16) when run under
    autotune with FLAGTREE_AABS=1.

Single-block GEMM (one program computes the whole MxN tile from MxK * KxN) keeps
the kernel simple and backend-portable; dims are multiples of the sunrise
min_dot_size (M=8,N=8,K=16/4).
"""

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
import pytest

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def gemm_local_ptr_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)

    # Load A (MxK) and B (KxN) from global memory.
    a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])

    # Stage A through a shared-memory buffer via local_ptr (the TLE path), then
    # read it back and feed tl.dot. This forces local store -> barrier -> load.
    a_smem = tle.gpu.alloc([M, K], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    row = tl.broadcast_to(tl.arange(0, M)[:, None], (M, K))
    col = tl.broadcast_to(tl.arange(0, K)[None, :], (M, K))
    a_ptrs = tle.gpu.local_ptr(a_smem, (row, col))
    tl.store(a_ptrs, a)
    a_staged = tl.load(a_ptrs)

    acc = tl.dot(a_staged, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)


@pytest.mark.parametrize("M,N,K", [(16, 16, 16), (32, 32, 32), (8, 8, 16)])
def test_gemm_local_ptr(M, N, K):
    torch.manual_seed(0)
    a = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
    c = torch.empty(M, N, device=DEVICE, dtype=torch.float32)

    gemm_local_ptr_kernel[(1, )](a, b, c, M, N, K)

    ref = a @ b
    torch.testing.assert_close(c.cpu(), ref.cpu(), atol=1e-2, rtol=1e-2)


@triton.jit
def gemm_plain_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
    acc = tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)


@pytest.mark.parametrize("M,N,K", [(16, 16, 16), (32, 64, 32)])
def test_gemm_plain(M, N, K):
    """Plain tl.dot (no local_ptr) baseline; confirms dot path itself works."""
    torch.manual_seed(1)
    a = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
    c = torch.empty(M, N, device=DEVICE, dtype=torch.float32)

    gemm_plain_kernel[(1, )](a, b, c, M, N, K)

    torch.testing.assert_close(c.cpu(), (a @ b).cpu(), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
