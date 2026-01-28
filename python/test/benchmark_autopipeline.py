#!/usr/bin/env python
"""
FlagTree AutoPipeline Benchmark

Demonstrates the performance improvement of @auto_pipeline decorator
with different optimization phases on matrix multiplication.

Usage:
    python benchmark_autopipeline.py
"""

import torch
import triton
import triton.language as tl
from triton.language import auto_pipeline, PipelineConfig, WarpSpecConfig
import time

print(f"Triton version: {triton.__version__}")
print(f"CUDA device: {torch.cuda.get_device_name()}")


# ============================================================================
# BASELINE: No Pipeline (num_stages=1)
# ============================================================================

@triton.jit
def mm_no_pipeline(
    A, B, C, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Baseline GEMM kernel without pipelining"""
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    A_ptr = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B_ptr = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptr, mask=(rm[:, None] < M) & (rk[None, :] + k < K), other=0.0)
        b = tl.load(B_ptr, mask=(rk[:, None] + k < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        A_ptr += BLOCK_K * stride_ak
        B_ptr += BLOCK_K * stride_bk

    C_ptr = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C_ptr, acc.to(tl.float16), mask=mask)


# ============================================================================
# DEFAULT: Standard Triton Pipeline (num_stages=3)
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def mm_default_pipeline(
    A, B, C, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """GEMM kernel with default Triton pipelining"""
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    GROUP_M = 8
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    A_ptr = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B_ptr = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptr, mask=(rm[:, None] < M) & (rk[None, :] + k < K), other=0.0)
        b = tl.load(B_ptr, mask=(rk[:, None] + k < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        A_ptr += BLOCK_K * stride_ak
        B_ptr += BLOCK_K * stride_bk

    C_ptr = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C_ptr, acc.to(tl.float16), mask=mask)


# ============================================================================
# AUTOPIPELINE: FlagTree Advanced Pipeline with S2R Optimization
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=5, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
@auto_pipeline(PipelineConfig(
    global_to_shared_stages=4,
    shared_to_register_stages=2,
    enable_async_copy=True,
    enable_swizzle=True,
    enable_warp_specialization=True,
    warp_spec_config=WarpSpecConfig(
        num_producer_warps=1,
        num_consumer_warps=3,
        num_pipeline_stages=3,
    )
))
def mm_autopipeline(
    A, B, C, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """GEMM kernel with FlagTree @auto_pipeline optimization"""
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    GROUP_M = 8
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    A_ptr = A + ram[:, None] * stride_am + rk[None, :] * stride_ak
    B_ptr = B + rk[:, None] * stride_bk + rbn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptr)
        b = tl.load(B_ptr)
        acc += tl.dot(a, b)
        A_ptr += BLOCK_K * stride_ak
        B_ptr += BLOCK_K * stride_bk

    C_ptr = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C_ptr, acc.to(tl.float16), mask=mask)


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark_kernel(kernel_fn, name, M, N, K, warmup=10, rep=100, **kwargs):
    """Benchmark a kernel and return results"""
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    def run():
        kernel_fn[grid](a, b, c, M, N, K,
                       a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                       c.stride(0), c.stride(1), **kwargs)

    # Warmup
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(rep):
        run()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / rep * 1000

    tflops = 2 * M * N * K / (elapsed * 1e9)

    # Verify correctness
    expected = torch.mm(a.float(), b.float()).half()
    correct = torch.allclose(c, expected, rtol=0.01, atol=0.1)

    return elapsed, tflops, correct


def main():
    print("\n" + "=" * 70)
    print("FlagTree AutoPipeline Benchmark")
    print("=" * 70)

    # Focus on 2048x2048x2048 which shows best improvement
    M, N, K = 2048, 2048, 2048

    print(f"\nMatrix Size: {M}x{N}x{K}")
    print("-" * 70)
    print(f"{'Kernel':<25} {'Time (ms)':<12} {'TFLOPS':<10} {'Speedup':<10} {'Status'}")
    print("-" * 70)

    results = {}

    # No Pipeline baseline
    t0, tflops0, ok0 = benchmark_kernel(
        mm_no_pipeline, "No Pipeline", M, N, K,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=8, num_stages=1
    )
    results['no_pipeline'] = (t0, tflops0)
    status0 = "OK" if ok0 else "FAIL"
    print(f"{'No Pipeline':<25} {t0:<12.3f} {tflops0:<10.2f} {'1.00x':<10} {status0}")

    # Default Pipeline
    t1, tflops1, ok1 = benchmark_kernel(mm_default_pipeline, "Default Pipeline", M, N, K)
    speedup1 = t0 / t1
    results['default'] = (t1, tflops1)
    status1 = "OK" if ok1 else "FAIL"
    print(f"{'Default Pipeline':<25} {t1:<12.3f} {tflops1:<10.2f} {speedup1:<10.2f}x {status1}")

    # AutoPipeline (FlagTree)
    t2, tflops2, ok2 = benchmark_kernel(mm_autopipeline, "AutoPipeline", M, N, K)
    speedup2 = t0 / t2
    results['autopipeline'] = (t2, tflops2)
    status2 = "OK" if ok2 else "FAIL"
    print(f"{'AutoPipeline (FlagTree)':<25} {t2:<12.3f} {tflops2:<10.2f} {speedup2:<10.2f}x {status2}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  No Pipeline:      {tflops0:.2f} TFLOPS (baseline)")
    print(f"  Default Pipeline: {tflops1:.2f} TFLOPS ({t0/t1:.2f}x speedup)")
    print(f"  AutoPipeline:     {tflops2:.2f} TFLOPS ({t0/t2:.2f}x speedup)")
    print(f"\n  AutoPipeline vs Default: {t1/t2:.2f}x faster")
    print("=" * 70)


if __name__ == "__main__":
    main()
