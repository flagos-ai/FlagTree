"""
Gluon SME matmul matching the fixed Triton config from 03-matrix-multiplication-autotune.py
(BLOCK_M/N=256, BLOCK_K=32, num_warps=16, num_stages=2).

TTGIR summary (fp16, EVEN_K):
  - A tile 256x32: SME blocked + async_copy stride=stride_am, useSme=1
  - B tile 32x256: SME blocked + async_copy stride=stride_bk, useSme=2
  - shared: swizzled_shared vec=32, useTcu=true, order=[1,0]
  - MMA: iluvatar_mma warpsPerCTA=[4,4], instrShape=[16,16,16]
  - software pipeline num_stages=2 (double-buffer A/B)

Bench:
  python3 third_party/iluvatar/python/tutorials/performance_test/bench_gluon_sme_matmul.py

ixkn-cli single-launch profile (skip correctness; 3 warmups then 1 launch):
  ixkn-cli -f -c 1 -s 3 --kernel-name "^gluon_sme_matmul_kernel$" --section all \\
    python3 third_party/iluvatar/python/tutorials/performance_test/bench_gluon_sme_matmul.py \\
      --profile-once --skip-check --kernel gluon
"""

from __future__ import annotations

import os

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon import iluvatar as gil
from triton.experimental.gluon.iluvatar.ivcore11 import async_copy as cp

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 32
NUM_WARPS = 16
NUM_STAGES = 2


@gluon.jit
def gluon_sme_matmul_kernel(
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
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    # Layouts mirror cache111 TTGIR encodings.
    a_sme: gl.constexpr = gil.IluvatarBlockedLayout([1, 8], [16, 4], [16, 1], [1, 0], is_sme=True,
                                                    sme_warps_per_cta=[16, 1])
    b_sme: gl.constexpr = gil.IluvatarBlockedLayout([1, 8], [16, 4], [2, 8], [1, 0], is_sme=True,
                                                    sme_warps_per_cta=[2, 8])
    shared: gl.constexpr = gil.IluvatarSwizzledSharedLayout(vec=32, per_phase=1, max_phase=1, order=[1, 0],
                                                            use_tcu=True)
    acc_layout: gl.constexpr = gil.IluvatarMMALayout(version=[1, 0], warps_per_cta=[4, 4], instr_shape=[16, 16, 16])
    lhs: gl.constexpr = gil.IluvatarDotOperandLayout(parent=acc_layout, operand_index=0, k_width=2, use_sme=1)
    rhs: gl.constexpr = gil.IluvatarDotOperandLayout(parent=acc_layout, operand_index=1, k_width=2, use_sme=2)
    # Match Triton's coalesced store layout from cache111 TTGIR (#linear).
    store_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [4, 0], [8, 0], [64, 0], [128, 0], [0, 128]],
        lane_bases=[[0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 0]],
        warp_bases=[[0, 32], [0, 64], [16, 0], [32, 0]],
        block_bases=[],
        shape=[BLOCK_M, BLOCK_N],
    )

    pid = gl.program_id(0)
    grid_n = gl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid - pid_m * grid_n

    offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, a_sme))
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, b_sme))
    offs_k_a = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, a_sme))
    offs_k_b = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, b_sme))

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k_a[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k_b[:, None] * stride_bk + offs_n[None, :] * stride_bn

    a_smem = gl.allocate_shared_memory(gl.float16, [2, BLOCK_M, BLOCK_K], shared)
    b_smem = gl.allocate_shared_memory(gl.float16, [2, BLOCK_K, BLOCK_N], shared)

    # Prefetch tile 0 (matches TTGIR prologue).
    cp.async_copy_global_to_shared(a_smem.index(0), a_ptrs, stride=stride_am)
    cp.commit_group()
    cp.async_copy_global_to_shared(b_smem.index(0), b_ptrs, stride=stride_bk)
    cp.commit_group()

    acc = gl.full([BLOCK_M, BLOCK_N], 0.0, gl.float32, acc_layout)
    num_k = gl.cdiv(K, BLOCK_K)

    # Even/odd unrolled ping-pong on memdesc.index (same packing as TTGIR 2x tiles).
    # The next tile's async copies must be issued *between* the shared-memory load and the mma
    for ki in range(0, num_k, 2):
        cp.wait_group(0)
        a_tile = a_smem.index(0).load(lhs)
        b_tile = b_smem.index(0).load(rhs)

        if ki + 1 < num_k:
            a_ptrs = a_ptrs + BLOCK_K * stride_ak
            b_ptrs = b_ptrs + BLOCK_K * stride_bk
            cp.async_copy_global_to_shared(a_smem.index(1), a_ptrs, stride=stride_am)
            cp.commit_group()
            cp.async_copy_global_to_shared(b_smem.index(1), b_ptrs, stride=stride_bk)
            cp.commit_group()

        acc = gil.ivcore11.mma(a_tile, b_tile, acc)

        if ki + 1 < num_k:
            cp.wait_group(0)
            a_tile = a_smem.index(1).load(lhs)
            b_tile = b_smem.index(1).load(rhs)

            if ki + 2 < num_k:
                a_ptrs = a_ptrs + BLOCK_K * stride_ak
                b_ptrs = b_ptrs + BLOCK_K * stride_bk
                cp.async_copy_global_to_shared(a_smem.index(0), a_ptrs, stride=stride_am)
                cp.commit_group()
                cp.async_copy_global_to_shared(b_smem.index(0), b_ptrs, stride=stride_bk)
                cp.commit_group()

            acc = gil.ivcore11.mma(a_tile, b_tile, acc)

    store_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, store_layout))
    store_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, store_layout))
    c_ptrs = c_ptr + store_m[:, None] * stride_cm + store_n[None, :] * stride_cn
    gl.store(c_ptrs, gl.convert_layout(acc.to(gl.float16), store_layout))


def gluon_sme_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    # Explicit double-buffering in-kernel; keep compiler stages=1.
    gluon_sme_matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps=NUM_WARPS,
        num_stages=1,
    )
    return c


def triton_fixed_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    import inspect
    from triton import cdiv
    import triton.language as tl
    from triton.ops.matmul import _kernel as triton_matmul_kernel

    jit_kernel = triton_matmul_kernel.fn.fn
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    meta = {
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "SPLIT_K": 1,
        "num_warps": NUM_WARPS,
        "num_stages": NUM_STAGES,
        "acc_dtype": tl.float32,
        "input_precision": None,
        "fp8_fast_accum": True,
        "GROUP_M": 8,
        "EVEN_K": True,
        "AB_DTYPE": tl.float16,
    }
    params = inspect.signature(jit_kernel.fn).parameters
    if "EVEN_M" in params:
        meta["EVEN_M"] = True
    if "EVEN_N" in params:
        meta["EVEN_N"] = True
    grid = lambda META: (cdiv(M, META["BLOCK_M"]) * cdiv(N, META["BLOCK_N"]), META["SPLIT_K"])
    jit_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        **meta,
    )
    return c


def _tflops(ms: float, M: int, N: int, K: int) -> float:
    return 2.0 * M * N * K * 1e-12 / (ms * 1e-3)


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Gluon vs Triton SME matmul bench / ixkn profile helper")
    p.add_argument("--size", type=int, default=int(os.getenv("MM_SIZE", "4096")))
    p.add_argument(
        "--profile-once",
        action="store_true",
        default=os.getenv("PROFILE_ONCE", "0") == "1",
        help="Warmup then a single kernel launch (for ixkn-cli). Skips do_bench loops.",
    )
    p.add_argument(
        "--kernel",
        choices=("gluon", "triton", "torch"),
        default=os.getenv("PROFILE_KERNEL", "gluon"),
        help="Which kernel to run in --profile-once mode (default: gluon).",
    )
    p.add_argument(
        "--skip-check",
        action="store_true",
        default=os.getenv("SKIP_CHECK", "0") == "1",
        help="Skip correctness check (faster for pure profiling).",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    M = N = K = args.size
    print(f"shape=({M},{N},{K}) config BLOCK=({BLOCK_M},{BLOCK_N},{BLOCK_K}) "
          f"warps={NUM_WARPS} stages={NUM_STAGES} profile_once={args.profile_once} kernel={args.kernel}")

    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    runners = {
        "torch": lambda: torch.matmul(a, b),
        "triton": lambda: triton_fixed_matmul(a, b),
        "gluon": lambda: gluon_sme_matmul(a, b),
    }

    if not args.skip_check:
        ref = runners["torch"]()
        out_triton = runners["triton"]()
        out_gluon = runners["gluon"]()
        torch.testing.assert_close(out_triton, ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out_gluon, ref, atol=1e-2, rtol=1e-2)
        print("correctness: OK (triton + gluon vs torch.matmul)")

    if args.profile_once:
        # Compile / warmup outside the profiled region when possible, then one launch.
        fn = runners[args.kernel]
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        print(f"profile-once: launching {args.kernel} once", flush=True)
        out = fn()
        torch.cuda.synchronize()
        print(f"profile-once: done, out.shape={tuple(out.shape)}", flush=True)
        return

    quantiles = [0.5, 0.2, 0.8]
    rows = []
    for name, key in [
        ("ixblas/torch", "torch"),
        ("triton-fixed", "triton"),
        ("gluon-sme", "gluon"),
    ]:
        ms, min_ms, max_ms = triton.testing.do_bench(runners[key], quantiles=quantiles)
        rows.append((name, _tflops(ms, M, N, K), _tflops(min_ms, M, N, K), _tflops(max_ms, M, N, K), ms))
        print(f"{name:14s}  median={_tflops(ms, M, N, K):8.3f} TFLOPS  "
              f"(min={_tflops(min_ms, M, N, K):8.3f}, max={_tflops(max_ms, M, N, K):8.3f})  ms={ms:.3f}")

    tri = next(r for r in rows if r[0] == "triton-fixed")
    glu = next(r for r in rows if r[0] == "gluon-sme")
    gap = (tri[1] - glu[1]) / tri[1] * 100.0
    print(f"gap gluon vs triton-fixed: {gap:+.2f}%  "
          f"(positive => gluon slower)")


if __name__ == "__main__":
    main()
