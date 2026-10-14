"""RDNA (gfx1201) bf16 WMMA GEMM microkernel for prefill (M>1).

FlagMega's dense_matmul is GEMV-only: it renders M>1 as a loop over
local_m_capacity with the weight load inside the m-loop, so it never uses the
matrix core -- ~3% of peak at M=128. For prefill (M = prompt length) a real
tl.dot GEMM reads each weight tile once and reuses it across BM rows via WMMA.

This is the tuned reference kernel + benchmark; it is the body intended for a
`tir.dense_matmul.mma` candidate (requires=("wmma",), preferred for M>1). RDNA
WMMA is reached purely through tl.dot on bf16 operands with fp32 accumulate
(no intrinsic), num_stages=1 ("GCN prefers shallow pipelines"), as in the
amd_radeon_kernels / FreeToken references.

Measured on R9700/gfx1201, C[M,2048] = A[M,2048] @ W[2048,2048]^T:
    M     FlagMega GEMV-loop    WMMA (best tile)
    32          309 us          34 us  ( 9x,  7.7 TFLOP/s)
    128         380 us          37 us  (10x, 29.0 TFLOP/s)   BM32 BN128 BK32
    512       ~1500 us          51 us  (~30x, 84.6 TFLOP/s)  BM64 BN256 BK64
84.6 TFLOP/s is ~89% of the ~95 TFLOP/s bf16 WMMA peak.
"""

import time

import torch
import triton
import triton.language as tl


@triton.jit
def _wmma_gemm(a_ptr, w_ptr, c_ptr, M, N, K,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """C[M, N] = A[M, K] @ W[N, K]^T, bf16 in / bf16 out, fp32 accumulate."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    a_p = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    w_p = w_ptr + offs_n[:, None] * K + offs_k[None, :]
    for k in range(0, K, BK):
        a = tl.load(a_p, mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0)
        w = tl.load(w_p, mask=(offs_n[:, None] < N) & ((k + offs_k)[None, :] < K), other=0.0)
        acc += tl.dot(a, tl.trans(w), out_dtype=tl.float32)  # WMMA on RDNA
        a_p += BK
        w_p += BK
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def wmma_gemm(a, w, *, BM=64, BN=256, BK=64, num_warps=8, num_stages=1):
    """a: [M, K] bf16, w: [N, K] bf16 (row-major) -> [M, N] bf16."""
    M, K = a.shape
    N = w.shape[0]
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _wmma_gemm[grid](a, w, c, M, N, K, BM, BN, BK, num_warps=num_warps, num_stages=num_stages)
    return c


def _benchmark():
    K = N = 2048
    w = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
    print("M      us     TFLOP/s   max_err")
    for M in (32, 128, 512):
        a = (torch.randn(M, K, device="cuda") * 0.02).to(torch.bfloat16)
        tile = {32: (32, 128, 32, 4), 128: (32, 128, 32, 4), 512: (64, 256, 64, 8)}[M]
        BM, BN, BK, nw = tile
        c = wmma_gemm(a, w, BM=BM, BN=BN, BK=BK, num_warps=nw)
        err = (c.float() - a.float() @ w.float().T).abs().max().item()
        for _ in range(10):
            wmma_gemm(a, w, BM=BM, BN=BN, BK=BK, num_warps=nw)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            wmma_gemm(a, w, BM=BM, BN=BN, BK=BK, num_warps=nw)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / 100 * 1e6
        print(f"{M:<5d} {us:7.1f} {2 * M * N * K / us / 1e6:8.1f}   {err:.3f}")


if __name__ == "__main__":
    _benchmark()
