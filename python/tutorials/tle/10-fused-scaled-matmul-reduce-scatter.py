"""Fused scaled matmul + reduce-scatter with FlagTree TLE.

This tutorial computes

    out = reduce_scatter(sum_r (A_r @ B_r), scatter_dim)

with a warp-specialized GEMM stage and a TLE reduce stage:

1. ``wgmma_gemm_kernel_2p`` runs one TMA producer per operand and a WGMMA
   consumer. The consumer applies the activation/weight scales and an optional
   bias, writes this rank's output tiles directly to ``out``, and writes the
   tiles owned by the peer into this rank's symmetric-memory window.
2. After a device barrier, ``reduce_scatter_out_kernel`` adds the peer's
   contributions (read with ``tle.remote(space="device")``) on top of the local
   tiles already in ``out``.

The window holds two buffers, so consecutive calls alternate buffers and no
trailing barrier is needed. Tile configurations are selected at runtime.
Every shape must be divisible by the chosen tile configuration (M/N/K and this
rank's output slice).

Run from the repository root on one node with at least two GPUs:

    NPROC_PER_NODE=2 M=1024 N=4096 K=2048 \
        bash python/tutorials/tle/10-fused-scaled-matmul-reduce-scatter.sh

M/N/K default to 1024/4096/2048. Use ``--scatter-dim 1`` to scatter columns
and ``--bias`` to add a bias. Requires SM90 or newer and matching
FlagTree/FlagCX runtime and device bitcode. Use CUDA_VISIBLE_DEVICES to select
the GPUs.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import statistics

import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.tools.tensor_descriptor import TensorDescriptor

TILE_CONFIGS = [
    dict(block_m=128, block_n=128, block_k=128, num_slots=2),
    dict(block_m=128, block_n=128, block_k=128, num_slots=4),
    dict(block_m=64, block_n=128, block_k=128, num_slots=2),
    dict(block_m=64, block_n=128, block_k=128, num_slots=4),
]

TUNE_WARMUP = 3
TUNE_ITERS = 10
WARMUP = 5
ITERS = 50


@triton.jit
def _wgmma_consumer(
    a_smem,
    b_smem,
    a_empty,
    b_empty,
    a_full,
    b_full,
    c_ptr,
    out_ptr,
    a_scale_ptr,
    b_scale_ptr,
    bias_ptr,
    M,
    N,
    SLOT_ELEMS,
    WINDOW_OFFSET,
    pid_m,
    pid_n,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_TILES: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SCATTER_DIM: tl.constexpr,
):
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    slot = 0
    phase = 0
    tle.gpu.barrier_wait(a_full[slot], phaseIdx=phase)
    tle.gpu.barrier_wait(b_full[slot], phaseIdx=phase)
    # 192 imprecise accumulations keep the error ~1%; exact accumulation is slower.
    acc = tle.gpu.wgmma(
        a_smem.slot(slot),
        b_smem.slot(slot),
        acc,
        trans_b=True,
        max_num_imprecise_acc=192,
    )
    last_slot = slot
    last_phase = phase

    for k_iter in range(1, K_TILES):
        slot = k_iter % NUM_SLOTS
        phase = k_iter // NUM_SLOTS
        tle.gpu.barrier_wait(a_full[slot], phaseIdx=phase)
        tle.gpu.barrier_wait(b_full[slot], phaseIdx=phase)
        acc = tle.gpu.wgmma(
            a_smem.slot(slot),
            b_smem.slot(slot),
            acc,
            trans_b=True,
            max_num_imprecise_acc=192,
        )
        acc = tle.gpu.wgmma_wait(1, acc)
        tle.gpu.barrier_arrive(a_empty[last_slot], phaseIdx=last_phase)
        tle.gpu.barrier_arrive(b_empty[last_slot], phaseIdx=last_phase)
        last_slot = slot
        last_phase = phase

    acc = tle.gpu.wgmma_wait(0, acc)
    tle.gpu.barrier_arrive(a_empty[last_slot], phaseIdx=last_phase)
    tle.gpu.barrier_arrive(b_empty[last_slot], phaseIdx=last_phase)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Per-row activation scale and per-column weight scale.
    acc *= tl.load(a_scale_ptr + offs_m).to(tl.float32)[:, None]
    acc *= tl.load(b_scale_ptr + offs_n).to(tl.float32)[None, :]
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n).to(tl.float32)[None, :]

    # Tiles this rank owns go straight to out; the peer's tiles go to the
    # window slot this rank owns, where the peer's reduce will pick them up.
    if SCATTER_DIM == 0:
        m_out = M // WORLD_SIZE
        owner = (pid_m * BLOCK_M) // m_out
        if owner == RANK:
            local_m = offs_m - RANK * m_out
            tl.store(
                out_ptr + local_m[:, None] * N + offs_n[None, :],
                acc.to(out_ptr.dtype.element_ty),
            )
        else:
            c_ptrs = (c_ptr + WINDOW_OFFSET + RANK * SLOT_ELEMS + offs_m[:, None] * N + offs_n[None, :])
            tl.store(c_ptrs, acc)
    else:
        n_out = N // WORLD_SIZE
        owner = (pid_n * BLOCK_N) // n_out
        if owner == RANK:
            local_n = offs_n - RANK * n_out
            tl.store(
                out_ptr + offs_m[:, None] * n_out + local_n[None, :],
                acc.to(out_ptr.dtype.element_ty),
            )
        else:
            c_ptrs = (c_ptr + WINDOW_OFFSET + RANK * SLOT_ELEMS + offs_m[:, None] * N + offs_n[None, :])
            tl.store(c_ptrs, acc)


@triton.jit
def _wgmma_producer(
    desc,
    smem,
    empty,
    full,
    pid,
    BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_TILES: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
):
    """Stream one operand's K tiles into NUM_SLOTS shared-memory slots."""
    for k_iter in range(0, K_TILES):
        slot = k_iter % NUM_SLOTS
        phase = k_iter // NUM_SLOTS
        tle.gpu.barrier_wait(empty[slot], phaseIdx=phase)
        tle.gpu.copy(
            desc,
            smem.slot(slot),
            [BLOCK, BLOCK_K],
            [pid * BLOCK, k_iter * BLOCK_K],
            barrier=full[slot],
        )
    for k_iter in range(K_TILES - NUM_SLOTS, K_TILES):
        slot = k_iter % NUM_SLOTS
        phase = k_iter // NUM_SLOTS
        tle.gpu.barrier_wait(empty[slot], phaseIdx=phase + 1)


@triton.jit
def wgmma_gemm_kernel_2p(
    a_desc,
    b_desc,
    c_ptr,
    out_ptr,
    a_scale_ptr,
    b_scale_ptr,
    bias_ptr,
    M,
    N,
    SLOT_ELEMS,
    WINDOW_OFFSET,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_TILES: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SCATTER_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    a_smem = tle.gpu.alloc(
        [NUM_SLOTS, BLOCK_M, BLOCK_K],
        dtype=tl.float8e4nv,
        layout=None,
        scope=tle.gpu.smem,
    )
    b_smem = tle.gpu.alloc(
        [NUM_SLOTS, BLOCK_N, BLOCK_K],
        dtype=tl.float8e4nv,
        layout=None,
        scope=tle.gpu.smem,
    )
    a_empty = tle.gpu.alloc_barriers(num_barriers=NUM_SLOTS, init=tle.gpu.READY)
    b_empty = tle.gpu.alloc_barriers(num_barriers=NUM_SLOTS, init=tle.gpu.READY)
    a_full = tle.gpu.alloc_barriers(num_barriers=NUM_SLOTS, expect_bytes=BLOCK_M * BLOCK_K)
    b_full = tle.gpu.alloc_barriers(num_barriers=NUM_SLOTS, expect_bytes=BLOCK_N * BLOCK_K)

    # One consumer (WGMMA + epilogue) and two TMA producers (A and B).
    tle.gpu.warp_specialize(
        [
            (
                _wgmma_consumer,
                (
                    a_smem,
                    b_smem,
                    a_empty,
                    b_empty,
                    a_full,
                    b_full,
                    c_ptr,
                    out_ptr,
                    a_scale_ptr,
                    b_scale_ptr,
                    bias_ptr,
                    M,
                    N,
                    SLOT_ELEMS,
                    WINDOW_OFFSET,
                    pid_m,
                    pid_n,
                    RANK,
                    WORLD_SIZE,
                    BLOCK_M,
                    BLOCK_N,
                    K_TILES,
                    NUM_SLOTS,
                    HAS_BIAS,
                    SCATTER_DIM,
                ),
            ),
            (
                _wgmma_producer,
                (
                    a_desc,
                    a_smem,
                    a_empty,
                    a_full,
                    pid_m,
                    BLOCK_M,
                    BLOCK_K,
                    K_TILES,
                    NUM_SLOTS,
                ),
            ),
            (
                _wgmma_producer,
                (
                    b_desc,
                    b_smem,
                    b_empty,
                    b_full,
                    pid_n,
                    BLOCK_N,
                    BLOCK_K,
                    K_TILES,
                    NUM_SLOTS,
                ),
            ),
        ],
        [1, 1],
        [48, 48],
    )


@triton.jit
def device_barrier_kernel(
    dist_ctx: tl.constexpr,
    mesh: tl.constexpr,
    INDEX: tl.constexpr,
):
    """Cross-rank device barrier; replaces the host-side dist.barrier()."""
    tle.distributed_barrier(
        mesh=mesh,
        device_dptr=dist_ctx,
        space="device",
        group_kind="block",
        barrier_kind="sync",
        order="acqrel",
        index=INDEX,
    )


@triton.jit
def reduce_scatter_out_kernel(
    dist_ctx: tl.constexpr,
    window_ptr,
    out_ptr,
    M,
    N,
    SLOT_ELEMS,
    WINDOW_OFFSET,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCATTER_DIM: tl.constexpr,
):
    """Add the peer contributions on top of the local tiles already in out."""
    if SCATTER_DIM == 0:
        M_out = M // WORLD_SIZE
        N_out = N
    else:
        M_out = M
        N_out = N // WORLD_SIZE

    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N_out, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for peer in tl.static_range(WORLD_SIZE):
        if peer != RANK:
            base = tle.remote(
                dist_ctx,
                space="device",
                dtype=window_ptr.dtype.element_ty,
                shard_id=peer,
                offset=0,
            )
            peer_slot = base + WINDOW_OFFSET + peer * SLOT_ELEMS
            if SCATTER_DIM == 0:
                global_m = RANK * M_out + offs_m
                val = tl.load(peer_slot + global_m[:, None] * N + offs_n[None, :])
            else:
                global_n = RANK * N_out + offs_n
                val = tl.load(peer_slot + offs_m[:, None] * N + global_n[None, :])
            acc += val.to(tl.float32)

    local = tl.load(out_ptr + offs_m[:, None] * N_out + offs_n[None, :])
    tl.store(
        out_ptr + offs_m[:, None] * N_out + offs_n[None, :],
        (local.to(tl.float32) + acc).to(out_ptr.dtype.element_ty),
    )


def set_tma_allocator():
    """TMA descriptors need a scratch allocator."""

    def _alloc(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(_alloc)


def _quantize_rows(x):
    amax = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
    scale = (amax / 448.0).to(torch.float32)
    return (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn), scale


def _quantize_cols(x):
    amax = x.abs().amax(dim=0, keepdim=True).clamp_min(1e-6)
    scale = (amax / 448.0).to(torch.float32)
    return (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn), scale


def _scaled_mm(A, B_col, A_scale, B_scale, bias):
    return torch._scaled_mm(A, B_col, A_scale, B_scale, bias, None, torch.bfloat16)


def _reference(A, B, A_scale, B_scale, bias, world_size, rank, scatter_dim):
    """Per-rank scaled matmul, cross-rank sum, then this rank's slice."""
    c = _scaled_mm(A, B.t().contiguous().t(), A_scale, B_scale, bias).float()
    dist.all_reduce(c, op=dist.ReduceOp.SUM)
    return c.chunk(world_size, dim=scatter_dim)[rank]


def _torch_baseline(A, B_col, A_scale, B_scale, bias, out, world_size, scatter_dim):
    """Eager torch path: scaled matmul followed by a collective."""
    c = _scaled_mm(A, B_col, A_scale, B_scale, bias)
    if scatter_dim == 0:
        dist.reduce_scatter_tensor(out, c, op=dist.ReduceOp.SUM)
    else:
        full = c.float()
        dist.all_reduce(full, op=dist.ReduceOp.SUM)
        out.copy_(full.chunk(world_size, dim=1)[dist.get_rank()])


def _original_baseline(A, B_col, A_scale, B_scale, bias, group_name, scatter_dim):
    """The symmetric-memory operator this tutorial reimplements."""
    shape = [A.shape[0], B_col.shape[1]]
    return torch.ops.symm_mem.fused_scaled_matmul_reduce_scatter(
        A,
        B_col,
        A_scale,
        B_scale,
        "sum",
        scatter_dim,
        scatter_dim,
        group_name,
        shape,
        bias,
        None,
        torch.bfloat16,
        False,
    )


def _time_ms(fn, warmup, iters, rounds=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iters)
        dist.barrier()
    local = torch.tensor([statistics.median(samples)], device="cuda", dtype=torch.float64)
    dist.all_reduce(local, op=dist.ReduceOp.MAX)
    return local.item()


def _check_on_all_ranks(actual, expected):
    want = expected.float()
    local_ok = torch.allclose(
        actual.float(),
        want,
        rtol=2e-2,
        atol=2e-2 * want.abs().max().clamp_min(1.0),
    )
    flag = torch.tensor(int(local_ok), dtype=torch.int32, device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if flag.item() == 0:
        max_err = (actual.float() - want).abs().max().item()
        raise AssertionError(f"reduce-scatter mismatch: max abs error {max_err:.4f}")


def _valid(config, M, N, K, M_out, N_out):
    return (M % config["block_m"] == 0 and N % config["block_n"] == 0 and K % config["block_k"] == 0
            and M_out % config["block_m"] == 0 and N_out % config["block_n"] == 0)


def _tune(make_run, candidates):
    best = None
    best_ms = float("inf")
    for config in candidates:
        ms = _time_ms(make_run(config), TUNE_WARMUP, TUNE_ITERS, rounds=2)
        if ms < best_ms:
            best, best_ms = config, ms
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--K", type=int, default=2048)
    parser.add_argument("--scatter-dim", type=int, choices=(0, 1), default=0)
    parser.add_argument("--bias", action="store_true")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"rank {local_rank} needs a GPU but only {torch.cuda.device_count()} "
                           f"are visible; set CUDA_VISIBLE_DEVICES to the {local_world_size} "
                           "GPUs you want to use")
    torch.cuda.set_device(local_rank)

    # get_mem_pool() initializes the process group and FlagCX, and returns the
    # allocator used for symmetric memory.
    mem_pool = tle.get_mem_pool()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    M, N, K = args.M, args.N, args.K
    scatter_dim = args.scatter_dim
    if world_size < 2 or world_size != local_world_size:
        raise ValueError("this tutorial requires one node with at least two GPUs")
    if torch.cuda.get_device_capability(local_rank)[0] < 9:
        raise ValueError("this tutorial requires SM90 or newer")
    if (M if scatter_dim == 0 else N) % world_size:
        raise ValueError("the scatter dimension must be divisible by world_size")

    M_out = M // world_size if scatter_dim == 0 else M
    N_out = N if scatter_dim == 0 else N // world_size
    candidates = [c for c in TILE_CONFIGS if _valid(c, M, N, K, M_out, N_out)]
    if not candidates:
        raise ValueError(f"no tile config divides M/N/K={M}/{N}/{K} and the output "
                         f"shape {M_out}x{N_out}; use larger aligned sizes")

    torch.manual_seed(1000 + rank)
    A_bf16 = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    B_bf16 = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
    A, A_scale = _quantize_rows(A_bf16)
    B, B_scale = _quantize_cols(B_bf16)
    B_t = B.t().contiguous()
    B_col = B.t().contiguous().t()
    bias = (torch.randn((N, ), device="cuda", dtype=torch.bfloat16) if args.bias else None)

    with torch.cuda.use_mem_pool(mem_pool):
        window = torch.empty((2 * world_size, M, N), device="cuda", dtype=torch.bfloat16)
    dist_ctx = tle.create_dist_tensor(window)
    mesh = tle.device_mesh(tle.MeshConfig(device=world_size))
    set_tma_allocator()

    # The symmetric-memory operator is the best baseline for row scatter.
    # Its column-scatter path does not handle this row-wise A_scale layout, so
    # fall back to the eager torch path there.
    baseline_name = "torch-eager"
    original_group = None
    if (scatter_dim == 0 and importlib.util.find_spec("torch.distributed._symmetric_memory") is not None):
        import torch.distributed._symmetric_memory as sm

        original_group = dist.distributed_c10d._get_default_group().group_name
        sm.enable_symm_mem_for_group(original_group)
        if hasattr(torch.ops.symm_mem, "fused_scaled_matmul_reduce_scatter"):
            baseline_name = "torch-symm-mem"

    out = torch.empty((M_out, N_out), device="cuda", dtype=torch.bfloat16)
    slot_elems = M * N
    buffer_stride = world_size * slot_elems

    def make_run(config):
        bm, bn, bk = config["block_m"], config["block_n"], config["block_k"]
        ns = config["num_slots"]
        a_desc = TensorDescriptor.from_tensor(A, block_shape=[bm, bk])
        b_desc = TensorDescriptor.from_tensor(B_t, block_shape=[bn, bk])
        offset = [0]

        def run():
            off = offset[0]
            offset[0] ^= buffer_stride
            wgmma_gemm_kernel_2p[(triton.cdiv(M, bm) * triton.cdiv(N, bn), )](
                a_desc,
                b_desc,
                window,
                out,
                A_scale,
                B_scale,
                bias if bias is not None else A,
                M,
                N,
                slot_elems,
                off,
                RANK=rank,
                WORLD_SIZE=world_size,
                BLOCK_M=bm,
                BLOCK_N=bn,
                BLOCK_K=bk,
                K_TILES=triton.cdiv(K, bk),
                NUM_SLOTS=ns,
                HAS_BIAS=bias is not None,
                SCATTER_DIM=scatter_dim,
                num_warps=4,
            )
            device_barrier_kernel[(1, )](dist_ctx, mesh, INDEX=0)
            reduce_scatter_out_kernel[(triton.cdiv(M_out, bm) * triton.cdiv(N_out, bn), )](
                dist_ctx,
                window,
                out,
                M,
                N,
                slot_elems,
                off,
                RANK=rank,
                WORLD_SIZE=world_size,
                BLOCK_M=bm,
                BLOCK_N=bn,
                SCATTER_DIM=scatter_dim,
                num_warps=8,
            )

        return run

    config = _tune(make_run, candidates)
    run_tle = make_run(config)

    run_tle()
    torch.cuda.synchronize()
    dist.barrier()

    expected = _reference(A, B, A_scale, B_scale, bias, world_size, rank, scatter_dim)
    _check_on_all_ranks(out, expected)

    if original_group is not None and baseline_name == "torch-symm-mem":

        def run_torch():
            _original_baseline(
                A,
                B_col,
                A_scale,
                B_scale,
                bias,
                original_group,
                scatter_dim,
            )

    else:

        def run_torch():
            _torch_baseline(A, B_col, A_scale, B_scale, bias, out, world_size, scatter_dim)

    tle_ms = _time_ms(run_tle, WARMUP, ITERS)
    torch_ms = _time_ms(run_torch, WARMUP, ITERS)
    tflops = 2 * M * N * K / (tle_ms * 1e-3) / 1e12

    if rank == 0:
        print(
            f"shape=({M}, {N}, {K}) world_size={world_size} "
            f"scatter_dim={scatter_dim} dtype=fp8 e4m3 bias={args.bias} "
            f"config={config}\n"
            f"  TLE    : {tle_ms:.3f} ms  ({tflops:.1f} TFLOPS/GPU)\n"
            f"  torch  : {torch_ms:.3f} ms\n"
            f"  speedup: {torch_ms / tle_ms:.2f}x",
            flush=True,
        )

    tle.cleanup_communicator()


if __name__ == "__main__":
    main()
