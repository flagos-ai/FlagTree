# flagtree tle
"""
Cluster GEMM with TLE Remote
============================

Compare two GEMM implementations:
- baseline Triton GEMM
- cluster GEMM that reuses A tile via `tle.remote` inside a 2-block cluster

Run example:
  python python/tutorials/tle/04-cluster-gemm.py --m 4096 --n 4096 --k 4096
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

BLOCK_CLUSTER_MESH = tle.device_mesh({"block_cluster": [("cluster_x", 2)]})


def _select_dot_k(bk: int) -> int:
    if bk % 32 == 0:
        return 32
    if bk % 16 == 0:
        return 16
    raise ValueError(f"BK must be divisible by 16 for tensor-core dot path, got BK={bk}")


def _select_remote_dot_k(bk: int) -> int:
    # Remote A path is currently limited by cluster-shared load/repack overhead
    # rather than pure MMA throughput. Smaller DOT_K improves overlap and reduces
    # per-step remote pointer traffic in practice.
    if bk % 16 == 0:
        return 16
    raise ValueError(f"BK must be divisible by 16 for remote dot path, got BK={bk}")


@triton.jit
def _triton_gemm_kernel(
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
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    DOT_K: tl.constexpr,
    USE_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BN)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        for ks in range(0, BK, DOT_K):
            k_idx = k0 + ks + tl.arange(0, DOT_K)
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak
            b_ptrs = b_ptr + k_idx[:, None] * stride_bk + offs_n[None, :] * stride_bn
            if USE_MASK:
                a_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
                b_mask = (k_idx[:, None] < K) & (offs_n[None, :] < N)
                a = tl.load(a_ptrs, mask=a_mask, other=0.0)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            else:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    if USE_MASK:
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)
    else:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


@triton.jit
def _cluster_remote_gemm_kernel(
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
    mesh: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    DOT_K: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    USE_MASK: tl.constexpr,
    A_SLOTS: tl.constexpr,
    USE_NV_MMA_SMEM_LAYOUT: tl.constexpr,
):
    pid = tl.program_id(0)
    cluster_rank = tle.shard_id(mesh, "cluster_x")
    cluster_id = pid // CLUSTER_SIZE

    num_pid_n = tl.cdiv(N, BN)
    num_pid_n_group = tl.cdiv(num_pid_n, CLUSTER_SIZE)
    pid_m = cluster_id // num_pid_n_group
    pid_ng = cluster_id % num_pid_n_group
    pid_n = pid_ng * CLUSTER_SIZE + cluster_rank

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_row_base = offs_m - pid_m * BM
    a_rows_full = tl.broadcast_to(a_row_base[:, None], (BM, BK))
    a_cols_full = tl.broadcast_to(tl.arange(0, BK)[None, :], (BM, BK))
    a_rows_t = tl.broadcast_to(a_row_base[None, :], (DOT_K, BM))
    a_buf = tle.gpu.alloc(
        [A_SLOTS, BM, BK],
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=USE_NV_MMA_SMEM_LAYOUT,
    )
    a_buf_remote = tle.remote(a_buf, 0, scope=mesh)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    # Prefetch the first K-tile before entering the main loop, then keep
    # one tile in flight. This reduces the per-iteration sync points from 2 to 1.
    slot0 = 0
    slot0_full = tl.zeros((BM, BK), dtype=tl.int32) + slot0
    if cluster_rank == 0:
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        if USE_MASK:
            a_mask_tile = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            a_tile = tl.load(a_ptrs, mask=a_mask_tile, other=0.0)
        else:
            a_tile = tl.load(a_ptrs)
        a_local_ptr_tile = tle.gpu.local_ptr(a_buf, (slot0_full, a_rows_full, a_cols_full))
        if USE_MASK:
            tl.store(a_local_ptr_tile, a_tile, mask=a_mask_tile)
        else:
            tl.store(a_local_ptr_tile, a_tile)

    tle.distributed_barrier(mesh)

    for k0 in range(0, K, BK):
        iter_idx = k0 // BK
        slot = iter_idx % A_SLOTS

        for ks in range(0, BK, DOT_K):
            k_local = ks + tl.arange(0, DOT_K)
            a_cols_t = tl.broadcast_to(k_local[:, None], (DOT_K, BM))
            slot_dot_t = tl.zeros((DOT_K, BM), dtype=tl.int32) + slot
            a_ptr_remote = tle.gpu.local_ptr(a_buf_remote, (slot_dot_t, a_rows_t, a_cols_t))
            if USE_MASK:
                a_mask_t = ((k0 + k_local)[:, None] < K) & (offs_m[None, :] < M)
                a = tl.trans(tl.load(a_ptr_remote, mask=a_mask_t, other=0.0))
            else:
                a = tl.trans(tl.load(a_ptr_remote))

            b_ptrs = b_ptr + (k0 + k_local)[:, None] * stride_bk + offs_n[None, :] * stride_bn
            if USE_MASK:
                b_mask = ((k0 + k_local)[:, None] < K) & (offs_n[None, :] < N)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            else:
                b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc)

        # With a single slot, producer and consumer share the same DSMEM tile.
        # Ensure all CTAs finish reading current tile before rank0 overwrites it.
        if A_SLOTS == 1:
            tle.distributed_barrier(mesh)

        next_k0 = k0 + BK
        has_next = next_k0 < K
        next_iter = iter_idx + 1
        next_slot = next_iter % A_SLOTS
        next_slot_full = tl.zeros((BM, BK), dtype=tl.int32) + next_slot
        if has_next and cluster_rank == 0:
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + (next_k0 + offs_k)[None, :] * stride_ak
            if USE_MASK:
                a_mask_tile = (offs_m[:, None] < M) & ((next_k0 + offs_k)[None, :] < K)
                a_tile = tl.load(a_ptrs, mask=a_mask_tile, other=0.0)
            else:
                a_tile = tl.load(a_ptrs)
            a_local_ptr_tile = tle.gpu.local_ptr(a_buf, (next_slot_full, a_rows_full, a_cols_full))
            if USE_MASK:
                tl.store(a_local_ptr_tile, a_tile, mask=a_mask_tile)
            else:
                tl.store(a_local_ptr_tile, a_tile)

        tle.distributed_barrier(mesh)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    if USE_MASK:
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)
    else:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


def _require_cluster_remote_support() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required")
    major, _minor = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError("cluster+remote path requires sm90+ (Hopper or newer)")


def _grid_triton(M: int, N: int, BM: int, BN: int) -> tuple[int]:
    return (triton.cdiv(M, BM) * triton.cdiv(N, BN), )


def _grid_cluster_remote(M: int, N: int, BM: int, BN: int, cluster_size: int = 2) -> tuple[int]:
    num_pid_n = triton.cdiv(N, BN)
    num_pid_n_group = triton.cdiv(num_pid_n, cluster_size)
    return (triton.cdiv(M, BM) * num_pid_n_group, )


def _run_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    bm: int,
    bn: int,
    bk: int,
    num_warps: int,
    num_stages: int,
) -> None:
    M, K = a.shape
    N = b.shape[1]
    dot_k = _select_dot_k(bk)
    use_mask = (M % bm != 0) or (N % bn != 0) or (K % bk != 0)
    _triton_gemm_kernel[_grid_triton(M, N, bm, bn)](
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
        BM=bm,
        BN=bn,
        BK=bk,
        DOT_K=dot_k,
        USE_MASK=use_mask,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _run_cluster_remote(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    bm: int,
    bn: int,
    bk: int,
    num_warps: int,
    num_stages: int,
) -> None:
    M, K = a.shape
    N = b.shape[1]
    dot_k = _select_remote_dot_k(bk)
    use_mask = (M % bm != 0) or (N % bn != 0) or (K % bk != 0)
    a_slots = 2
    # BK=64 with 2-stage pipeline benefits from MMA-friendly shared layout in
    # the remote A path; BK=32 keeps the existing fast path.
    use_nv_mma_smem_layout = (bk == 32) or (bk == 64 and num_stages <= 2)
    _cluster_remote_gemm_kernel[_grid_cluster_remote(M, N, bm, bn)](
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
        mesh=BLOCK_CLUSTER_MESH,
        BM=bm,
        BN=bn,
        BK=bk,
        DOT_K=dot_k,
        CLUSTER_SIZE=2,
        USE_MASK=use_mask,
        A_SLOTS=a_slots,
        USE_NV_MMA_SMEM_LAYOUT=use_nv_mma_smem_layout,
        num_ctas=1,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _verify_remote_lowering(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    bm: int,
    bn: int,
    bk: int,
    num_warps: int,
    num_stages: int,
) -> None:
    M, K = a.shape
    N = b.shape[1]
    dot_k = _select_remote_dot_k(bk)
    use_mask = (M % bm != 0) or (N % bn != 0) or (K % bk != 0)
    # Keep verifier config aligned with runtime launch config.
    a_slots = 2
    use_nv_mma_smem_layout = (bk == 32) or (bk == 64 and num_stages <= 2)
    compiled = _cluster_remote_gemm_kernel.warmup(
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
        mesh=BLOCK_CLUSTER_MESH,
        BM=bm,
        BN=bn,
        BK=bk,
        DOT_K=dot_k,
        CLUSTER_SIZE=2,
        USE_MASK=use_mask,
        A_SLOTS=a_slots,
        USE_NV_MMA_SMEM_LAYOUT=use_nv_mma_smem_layout,
        grid=_grid_cluster_remote(M, N, bm, bn),
        num_ctas=1,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    cluster_dims = tuple(compiled.metadata.cluster_dims)
    if cluster_dims != (2, 1, 1):
        raise RuntimeError(f"unexpected cluster_dims={cluster_dims}, expect (2, 1, 1)")
    ptx = compiled.asm.get("ptx", "")
    ttgir = compiled.asm.get("ttgir", "")
    has_remote = (("mapa.shared::cluster" in ptx) or ("tle.remote_pointers" in ttgir) or ("tle.remote_cta_id" in ttgir)
                  or ("tle.remote_shard_id_carrier" in ttgir))
    if not has_remote:
        raise RuntimeError("remote lowering evidence not found in PTX/TTGIR")


def _tflops(M: int, N: int, K: int, ms: float) -> float:
    return 2.0 * M * N * K * 1e-12 / (ms * 1e-3)


def _print_ir_stats(name: str, compiled) -> None:
    ttgir = compiled.asm.get("ttgir", "")
    ptx = compiled.asm.get("ptx", "")
    ttgir_patterns = {
        "tt.dot": "tt.dot ",
        "tt.load": "tt.load ",
        "tt.store": "tt.store ",
        "ttg.local_alloc": "ttg.local_alloc",
        "ttg.local_load": "ttg.local_load",
        "ttg.convert_layout": "ttg.convert_layout",
        "tle.local_pointers": "\"tle.local_pointers\"",
        "tle.remote_pointers": "\"tle.remote_pointers\"",
        "tle.distributed_barrier": "tle.distributed_barrier",
    }
    ptx_patterns = {
        "mapa.shared::cluster": "mapa.shared::cluster",
        "ld.shared::cluster.b16": "ld.shared::cluster.b16",
        "cp.async": "cp.async",
        "ldmatrix.sync": "ldmatrix.sync",
        "mma.sync": "mma.sync",
        "barrier.cluster": "barrier.cluster",
    }
    print(f"\n[ir] {name}")
    print("  ttgir:")
    for k, p in ttgir_patterns.items():
        print(f"    {k:<26} {ttgir.count(p)}")
    print("  ptx:")
    for k, p in ptx_patterns.items():
        print(f"    {k:<26} {ptx.count(p)}")


@dataclass
class BenchResult:
    name: str
    ms: float
    min_ms: float
    max_ms: float
    tflops: float


@dataclass(frozen=True)
class KernelConfig:
    bm: int
    bn: int
    bk: int
    num_warps: int
    num_stages: int


TRITON_TUNE_CONFIGS = [
    KernelConfig(64, 256, 32, 8, 3),
    KernelConfig(64, 256, 64, 8, 2),
    KernelConfig(128, 128, 32, 8, 3),
]

REMOTE_TUNE_CONFIGS = [
    # Best observed on H100 for large square GEMM in current remote path.
    KernelConfig(32, 512, 256, 4, 2),
    KernelConfig(32, 512, 128, 4, 3),
    KernelConfig(32, 512, 128, 4, 2),
    KernelConfig(32, 512, 128, 8, 2),
    KernelConfig(32, 512, 64, 8, 2),
    KernelConfig(32, 512, 64, 8, 3),
    KernelConfig(32, 512, 32, 8, 3),
    KernelConfig(32, 512, 32, 8, 2),
    KernelConfig(32, 512, 32, 4, 2),
    # Wider N tile helps amortize remote A fetch overhead in cluster path.
    KernelConfig(64, 512, 32, 8, 2),
    KernelConfig(64, 512, 32, 8, 3),
    KernelConfig(32, 256, 32, 8, 3),
    KernelConfig(64, 256, 32, 8, 3),
    KernelConfig(64, 256, 32, 4, 3),
    KernelConfig(64, 256, 32, 4, 2),
    KernelConfig(64, 256, 64, 8, 2),
]


def _default_remote_config_for_no_autotune(
    triton_cfg: KernelConfig,
    bm: int,
    bn: int,
    bk: int,
    num_warps: int,
    num_stages: int,
) -> KernelConfig:
    # Keep explicit user-provided launch config unchanged.
    if (bm, bn, bk, num_warps, num_stages) != (64, 256, 32, 8, 2):
        return triton_cfg
    # For out-of-box run, prefer a remote-friendly tile observed to sustain
    # higher throughput after TTGIR-guided remote A path reshaping.
    return KernelConfig(64, 256, 64, 8, 2)


def _autotune_config(
    name: str,
    candidates: list[KernelConfig],
    run_fn: Callable[[KernelConfig], None],
    M: int,
    N: int,
    K: int,
    warmup: int,
    rep: int,
) -> KernelConfig:
    best_cfg = candidates[0]
    best_ms = float("inf")
    for cfg in candidates:
        run_fn(cfg)
        torch.cuda.synchronize()
        ms = triton.testing.do_bench(lambda: run_fn(cfg), warmup=warmup, rep=rep)
        if ms < best_ms:
            best_ms = ms
            best_cfg = cfg
    print(f"[autotune] {name}: best cfg={best_cfg} ms={best_ms:.3f} tflops={_tflops(M, N, K, best_ms):.2f}")
    return best_cfg


def _bench_provider(name: str, fn: Callable[[], None], M: int, N: int, K: int, warmup: int, rep: int) -> BenchResult:
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=quantiles)
    return BenchResult(name=name, ms=ms, min_ms=min_ms, max_ms=max_ms, tflops=_tflops(M, N, K, ms))


def _print_results(results: list[BenchResult]) -> None:
    base = results[0].ms
    print()
    print(f"{'provider':<24}{'ms':>10}{'tflops':>12}{'speedup':>12}")
    for r in results:
        speedup = base / r.ms
        print(f"{r.name:<24}{r.ms:>10.3f}{r.tflops:>12.2f}{speedup:>12.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--bm", type=int, default=64)
    parser.add_argument("--bn", type=int, default=256)
    parser.add_argument("--bk", type=int, default=32)
    parser.add_argument("--num-warps", type=int, default=8)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-lowering", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--autotune", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tune-warmup", type=int, default=10)
    parser.add_argument("--tune-rep", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--analyze-ir", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    _require_cluster_remote_support()

    torch.manual_seed(args.seed)
    dtype = torch.float16
    a = torch.randn((args.m, args.k), device="cuda", dtype=dtype)
    b = torch.randn((args.k, args.n), device="cuda", dtype=dtype)
    c_triton = torch.empty((args.m, args.n), device="cuda", dtype=dtype)
    c_remote = torch.empty_like(c_triton)

    triton_cfg = KernelConfig(args.bm, args.bn, args.bk, args.num_warps, args.num_stages)
    remote_cfg = _default_remote_config_for_no_autotune(
        triton_cfg,
        args.bm,
        args.bn,
        args.bk,
        args.num_warps,
        args.num_stages,
    )

    if args.autotune:
        triton_cfg = _autotune_config(
            "triton_gemm",
            TRITON_TUNE_CONFIGS,
            lambda cfg: _run_triton(a, b, c_triton, cfg.bm, cfg.bn, cfg.bk, cfg.num_warps, cfg.num_stages),
            args.m,
            args.n,
            args.k,
            args.tune_warmup,
            args.tune_rep,
        )
        remote_cfg = _autotune_config(
            "cluster_tle_remote_gemm",
            REMOTE_TUNE_CONFIGS,
            lambda cfg: _run_cluster_remote(
                a,
                b,
                c_remote,
                cfg.bm,
                cfg.bn,
                cfg.bk,
                cfg.num_warps,
                cfg.num_stages,
            ),
            args.m,
            args.n,
            args.k,
            args.tune_warmup,
            args.tune_rep,
        )

    print(f"selected triton cfg: {triton_cfg}")
    print(f"selected remote cfg: {remote_cfg}")

    if args.analyze_ir:
        triton_compiled = _triton_gemm_kernel.warmup(
            a,
            b,
            c_triton,
            args.m,
            args.n,
            args.k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c_triton.stride(0),
            c_triton.stride(1),
            BM=triton_cfg.bm,
            BN=triton_cfg.bn,
            BK=triton_cfg.bk,
            DOT_K=_select_dot_k(triton_cfg.bk),
            USE_MASK=((args.m % triton_cfg.bm != 0) or (args.n % triton_cfg.bn != 0) or (args.k % triton_cfg.bk != 0)),
            grid=_grid_triton(args.m, args.n, triton_cfg.bm, triton_cfg.bn),
            num_warps=triton_cfg.num_warps,
            num_stages=triton_cfg.num_stages,
        )
        remote_compiled = _cluster_remote_gemm_kernel.warmup(
            a,
            b,
            c_remote,
            args.m,
            args.n,
            args.k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c_remote.stride(0),
            c_remote.stride(1),
            mesh=BLOCK_CLUSTER_MESH,
            BM=remote_cfg.bm,
            BN=remote_cfg.bn,
            BK=remote_cfg.bk,
            DOT_K=_select_remote_dot_k(remote_cfg.bk),
            CLUSTER_SIZE=2,
            USE_MASK=((args.m % remote_cfg.bm != 0) or (args.n % remote_cfg.bn != 0) or (args.k % remote_cfg.bk != 0)),
            A_SLOTS=2,
            USE_NV_MMA_SMEM_LAYOUT=((remote_cfg.bk == 32) or (remote_cfg.bk == 64 and remote_cfg.num_stages <= 2)),
            grid=_grid_cluster_remote(args.m, args.n, remote_cfg.bm, remote_cfg.bn),
            num_ctas=1,
            num_warps=remote_cfg.num_warps,
            num_stages=remote_cfg.num_stages,
        )
        _print_ir_stats("triton_gemm", triton_compiled)
        _print_ir_stats("cluster_tle_remote_gemm", remote_compiled)

    if args.check_lowering:
        _verify_remote_lowering(
            a,
            b,
            c_remote,
            remote_cfg.bm,
            remote_cfg.bn,
            remote_cfg.bk,
            remote_cfg.num_warps,
            remote_cfg.num_stages,
        )
        print("remote lowering check: PASS")

    if args.check:
        _run_triton(a, b, c_triton, triton_cfg.bm, triton_cfg.bn, triton_cfg.bk, triton_cfg.num_warps,
                    triton_cfg.num_stages)
        _run_cluster_remote(
            a,
            b,
            c_remote,
            remote_cfg.bm,
            remote_cfg.bn,
            remote_cfg.bk,
            remote_cfg.num_warps,
            remote_cfg.num_stages,
        )
        ref = torch.matmul(a, b)
        torch.testing.assert_close(c_triton, ref, atol=1e-1, rtol=1e-1)
        torch.testing.assert_close(c_remote, ref, atol=1e-1, rtol=1e-1)
        print("correctness check: PASS")

    run_triton = lambda: _run_triton(a, b, c_triton, triton_cfg.bm, triton_cfg.bn, triton_cfg.bk, triton_cfg.num_warps,
                                     triton_cfg.num_stages)
    run_remote = lambda: _run_cluster_remote(
        a,
        b,
        c_remote,
        remote_cfg.bm,
        remote_cfg.bn,
        remote_cfg.bk,
        remote_cfg.num_warps,
        remote_cfg.num_stages,
    )

    results = [
        _bench_provider("triton_gemm", run_triton, args.m, args.n, args.k, args.warmup, args.rep),
        _bench_provider("cluster_tle_remote_gemm", run_remote, args.m, args.n, args.k, args.warmup, args.rep),
    ]
    _print_results(results)


if __name__ == "__main__":
    main()
