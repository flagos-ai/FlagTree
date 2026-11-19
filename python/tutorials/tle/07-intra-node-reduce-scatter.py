"""Function-level autotuning for intra-node reduce-scatter with TMA reduction.

Run from the repository root on one node with at least two GPUs:

    NPROC_PER_NODE=4 M=2048 N=2048 DTYPE=bf16 \
        bash python/tutorials/tle/07-intra-node-reduce-scatter.sh

MASTER_ADDR defaults to localhost; MASTER_PORT defaults to 29501.

M/N default to 2048/2048; DTYPE accepts bf16 or fp16 (default: bf16).
NPROC_PER_NODE defaults to 4; use CUDA_VISIBLE_DEVICES to select GPUs.
Requires SM90 or newer and matching FlagTree/FlagCX runtime and device bitcode.
Set FLAGCX_IB_HCA to the interfaces on your machine when needed.

The complete scatter -> device barrier -> TMA reduction is decorated with
flagtree.tune.autotune. It jointly tunes both kernels' tiles and warp counts,
plus scatter concurrency. All ranks call with autotune_pg=dist.group.WORLD
so they benchmark the same candidates and minimize the maximum rank latency.
An entry barrier makes repeated calls safe for buffer reuse.

CONFIG_SPACE generates independent tile, warp and grid choices. prune_fn
filters shapes, TMA alignment, resources and redundant grids, then compiles
locally without launching. The distributed tuner intersects the surviving
candidates before any rank executes communication kernels.

The first decorated call tunes on a cache miss; a second call with changed
data verifies cache reuse. Autotune owns result persistence, controlled by
FLAGTREE_AUTOTUNE_CACHE_DIR. Local compile checks are memoized in the context;
a new process still preflights candidates before consulting the tuning cache.
TLE_CLEAR_L2 controls only the final TLE/Torch benchmark printed to stdout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import statistics

import torch
import torch.distributed as dist

import triton
import triton.language as tl
import triton.runtime
import triton.experimental.tle.language as tle

from flagtree.tune import autotune, to_hashable


@triton.jit
def scatter_kernel_opt(
    input_ptr,
    local_scatter_ptr,
    scatter_ctx: tl.constexpr,
    M_per_rank,
    N,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    SCATTER_BLOCK_M: tl.constexpr,
    SCATTER_BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)

    num_tiles_m = tl.cdiv(M_per_rank, SCATTER_BLOCK_M)
    num_tiles_n = tl.cdiv(N, SCATTER_BLOCK_N)
    tiles_per_peer = num_tiles_m * num_tiles_n

    row_offs = tl.arange(0, SCATTER_BLOCK_M)
    col_offs = tl.arange(0, SCATTER_BLOCK_N)

    slot_offset_elems = LOCAL_RANK * M_per_rank * N

    for step in range(WORLD_SIZE):
        peer = (LOCAL_RANK + step + 1) % WORLD_SIZE

        if peer == LOCAL_RANK:
            remote_base = local_scatter_ptr + slot_offset_elems
        else:
            remote_base = tle.remote(
                scatter_ctx,
                space="device",
                dtype=input_ptr.dtype.element_ty,
                shard_id=peer,
                offset=slot_offset_elems,
            )

        for local_tile in range(pid, tiles_per_peer, num_pid):
            tile_m = local_tile // num_tiles_n
            tile_n = local_tile % num_tiles_n

            in_row = peer * M_per_rank + tile_m * SCATTER_BLOCK_M
            in_col = tile_n * SCATTER_BLOCK_N
            in_ptrs = (input_ptr + (in_row + row_offs[:, None]) * N + (in_col + col_offs[None, :]))

            in_row_mask = (in_row + row_offs[:, None]) < (peer + 1) * M_per_rank
            in_col_mask = (in_col + col_offs[None, :]) < N
            data = tl.load(in_ptrs, mask=in_row_mask & in_col_mask, other=0.0)

            out_row_in_peer = tile_m * SCATTER_BLOCK_M
            out_col = tile_n * SCATTER_BLOCK_N
            out_ptrs = (remote_base + (out_row_in_peer + row_offs[:, None]) * N + (out_col + col_offs[None, :]))
            out_row_mask = (out_row_in_peer + row_offs[:, None]) < M_per_rank
            tl.store(out_ptrs, data, mask=out_row_mask & in_col_mask)


@triton.jit
def ring_reduce_kernel_tma(
    local_scatter_ptr,
    output_ptr,
    M_per_rank,
    N,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    REDUCE_BLOCK_M: tl.constexpr,
    REDUCE_BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)

    num_tiles_m = tl.cdiv(M_per_rank, REDUCE_BLOCK_M)
    num_tiles_n = tl.cdiv(N, REDUCE_BLOCK_N)
    total_tiles = num_tiles_m * num_tiles_n

    c_desc = tl.make_tensor_descriptor(
        local_scatter_ptr,
        shape=[M_per_rank * WORLD_SIZE, N],
        strides=[N, 1],
        block_shape=[REDUCE_BLOCK_M, REDUCE_BLOCK_N],
    )

    output_desc = tl.make_tensor_descriptor(
        output_ptr,
        shape=[M_per_rank, N],
        strides=[N, 1],
        block_shape=[REDUCE_BLOCK_M, REDUCE_BLOCK_N],
    )

    begin_idx = LOCAL_RANK

    for tile_id in range(pid, total_tiles, num_pid):
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id % num_tiles_n

        row_in_shard = tile_m * REDUCE_BLOCK_M
        col = tile_n * REDUCE_BLOCK_N

        src_rank = (begin_idx + 1) % WORLD_SIZE
        accum = c_desc.load([
            row_in_shard + src_rank * M_per_rank,
            col,
        ])

        for i in range(1, WORLD_SIZE):
            src_rank = (i + begin_idx + 1) % WORLD_SIZE
            data = c_desc.load([
                row_in_shard + src_rank * M_per_rank,
                col,
            ])
            accum += data

        output_desc.store([row_in_shard, col], accum)


@triton.jit
def device_barrier_kernel(
    scatter_ctx: tl.constexpr,
    mesh: tl.constexpr,
):
    tle.distributed_barrier(
        mesh=mesh,
        device_dptr=scatter_ctx,
        space="device",
        group_kind="block",
        barrier_kind="sync",
        order="acqrel",
        memory_scope="system",
        index=0,
        context_id=0,
    )


def _get_problem():
    """Parse input dimensions and dtype."""
    M = int(os.environ.get("M", "2048"))
    N = int(os.environ.get("N", "2048"))
    if M <= 0 or N <= 0:
        raise ValueError("M and N must be positive integers")
    dtype_name = os.environ.get("DTYPE", "bf16").strip().lower()
    dtypes = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    if dtype_name not in dtypes:
        raise ValueError(f"unsupported DTYPE={dtype_name!r}; use bf16 or fp16")
    return M, N, dtypes[dtype_name]


# Generate the search space independently; all workload-dependent and
# search-budget constraints live in _prune_reduce_scatter_config.
CONFIG_SPACE = [{
    "config": {
        "SCATTER_BLOCK_M": sm,
        "SCATTER_BLOCK_N": sn,
        "REDUCE_BLOCK_M": rm,
        "REDUCE_BLOCK_N": rn,
        "scatter_num_warps": scatter_warps,
        "reduce_num_warps": reduce_warps,
        "scatter_grid_factor": grid_factor,
    }
}
                for sm in (64, 128, 256)
                for sn in (64, 128, 256)
                for rm in (64, 128, 256)
                for rn in (64, 128, 256)
                for scatter_warps in (4, 8)
                for reduce_warps in (4, 8)
                for grid_factor in (1, 2, 4)]

INPUT_SEED = 0
WARMUP_ITERS = 10
BENCH_ITERS = 200


def _grid_ctas(M_per_rank, N, block_m, block_n, num_sms, grid_factor=1):
    """Cap the grid at grid_factor * num_sms; -1 uses the full tile grid."""
    if num_sms != -1 and num_sms < 1:
        raise ValueError("num_sms must be positive or -1 for an uncapped grid")
    if grid_factor < 1:
        raise ValueError("grid_factor must be positive")
    tiles = triton.cdiv(M_per_rank, block_m) * triton.cdiv(N, block_n)
    return tiles if num_sms == -1 else min(tiles, grid_factor * num_sms)


@dataclass
class ReduceScatterContext:
    scatter_buf: torch.Tensor
    scatter_ctx: object
    output: torch.Tensor
    M_per_rank: int
    N: int
    local_rank: int
    world_size: int
    num_sms: int
    max_shared_memory: int
    calls: int = 0
    last_config: dict | None = None
    preflight_results: dict = field(default_factory=dict, repr=False)


def _launch_reduce_scatter(input_tensor, ctx, *, config, compile_only=False):
    """Entry barrier -> scatter -> barrier -> TMA reduction on the current stream."""
    mesh = tle.device_mesh(tle.MeshConfig(device=ctx.world_size))
    # A fast rank must not overwrite slots that a peer is still reducing
    # from the preceding call. The tuner enqueues calls back-to-back.
    entry = device_barrier_kernel.run(
        grid=(1, ),
        warmup=compile_only,
        scatter_ctx=ctx.scatter_ctx,
        mesh=mesh,
    )
    scatter = scatter_kernel_opt.run(
        input_tensor,
        ctx.scatter_buf,
        grid=(_grid_ctas(ctx.M_per_rank, ctx.N, config["SCATTER_BLOCK_M"], config["SCATTER_BLOCK_N"], ctx.num_sms,
                         config["scatter_grid_factor"]), ),
        warmup=compile_only,
        scatter_ctx=ctx.scatter_ctx,
        M_per_rank=ctx.M_per_rank,
        N=ctx.N,
        LOCAL_RANK=ctx.local_rank,
        WORLD_SIZE=ctx.world_size,
        SCATTER_BLOCK_M=config["SCATTER_BLOCK_M"],
        SCATTER_BLOCK_N=config["SCATTER_BLOCK_N"],
        num_warps=config["scatter_num_warps"],
    )
    barrier = device_barrier_kernel.run(
        grid=(1, ),
        warmup=compile_only,
        scatter_ctx=ctx.scatter_ctx,
        mesh=mesh,
    )
    reduce = ring_reduce_kernel_tma.run(
        ctx.scatter_buf,
        ctx.output,
        ctx.M_per_rank,
        ctx.N,
        grid=(_grid_ctas(ctx.M_per_rank, ctx.N, config["REDUCE_BLOCK_M"], config["REDUCE_BLOCK_N"], ctx.num_sms), ),
        warmup=compile_only,
        LOCAL_RANK=ctx.local_rank,
        WORLD_SIZE=ctx.world_size,
        REDUCE_BLOCK_M=config["REDUCE_BLOCK_M"],
        REDUCE_BLOCK_N=config["REDUCE_BLOCK_N"],
        num_warps=config["reduce_num_warps"],
    )
    if compile_only:
        # Resolve asynchronous compilation and check actual resource limits
        # by loading binaries. warmup=True never executes the kernels.
        for kernel in (entry, scatter, barrier, reduce):
            if hasattr(kernel, "result"):
                kernel = kernel.result()
            kernel._init_handles()


def _reduce_scatter_key(input_tensor, ctx):
    # Values, addresses, rank IDs and mutable state do not affect the workload.
    return (
        to_hashable(input_tensor),
        to_hashable(ctx.scatter_buf),
        to_hashable(ctx.output),
        ctx.world_size,
    )


def _prune_reduce_scatter_config(config, input_tensor, ctx):
    """Keep locally feasible candidates; autotune intersects them across ranks."""
    cfg = config["config"]
    sm, sn = cfg["SCATTER_BLOCK_M"], cfg["SCATTER_BLOCK_N"]
    rm, rn = cfg["REDUCE_BLOCK_M"], cfg["REDUCE_BLOCK_N"]
    if not (min(ctx.M_per_rank, ctx.N) > 0 and tuple(input_tensor.shape) == tuple(ctx.scatter_buf.shape) ==
            (ctx.M_per_rank * ctx.world_size, ctx.N) and tuple(ctx.output.shape) == (ctx.M_per_rank, ctx.N)
            and input_tensor.dtype == ctx.scatter_buf.dtype == ctx.output.dtype and input_tensor.dtype
            in (torch.float16, torch.bfloat16) and input_tensor.device == ctx.scatter_buf.device == ctx.output.device
            and all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in (input_tensor, ctx.scatter_buf, ctx.output))
            and max(sm, rm) <= ctx.M_per_rank and max(sn, rn) <= ctx.N
            # A TMA tile must not cross a source rank's row boundary.
            and ctx.M_per_rank % rm == 0 and (ctx.N * input_tensor.element_size()) % 16 == 0):
        return False

    # Tutorial search-budget heuristics, not hardware validity requirements:
    # explore 8K..32K-element tiles, with 8 warps only for the largest tiles.
    for elements, warps in ((sm * sn, cfg["scatter_num_warps"]), (rm * rn, cfg["reduce_num_warps"])):
        if not 8192 <= elements <= 32768 or (warps == 8 and elements < 32768):
            return False

    factor = cfg["scatter_grid_factor"]
    if factor > 1 and _grid_ctas(ctx.M_per_rank, ctx.N, sm, sn, ctx.num_sms, factor) == _grid_ctas(
            ctx.M_per_rank, ctx.N, sm, sn, ctx.num_sms, factor // 2):
        return False  # The tile count already caps both grids at the same size.

    # Pruning runs even on a tuning-cache hit. Memoize compile/load verdicts
    # for this context and workload. Grid size is not a JIT specialization.
    preflight_key = (
        _reduce_scatter_key(input_tensor, ctx),
        ctx.local_rank,
        str(input_tensor.device),
        id(ctx.scatter_ctx),
        tuple(sorted((name, value) for name, value in cfg.items() if name != "scatter_grid_factor")),
    )
    if preflight_key not in ctx.preflight_results:
        try:
            _launch_reduce_scatter(input_tensor, ctx, config=cfg, compile_only=True)
        except Exception as exc:
            ctx.preflight_results[preflight_key] = False
            print(f"[Rank {ctx.local_rank}] Pruned {cfg}: {type(exc).__name__}: {exc}", flush=True)
        else:
            ctx.preflight_results[preflight_key] = True
    # No collectives or execution here: ranks can prune different candidates.
    # The tuner's intersection excludes any candidate that failed on a peer.
    return ctx.preflight_results[preflight_key]


@autotune(
    config_space=CONFIG_SPACE,
    key_fn=_reduce_scatter_key,
    cache_version="tle-intra-node-rs-pruned-space-v1",
)
def reduce_scatter(input_tensor: torch.Tensor, ctx: ReduceScatterContext, *, config: dict) -> torch.Tensor:
    """Tune and execute on the current stream; all ranks pass autotune_pg=dist.group.WORLD.

    Buffers must be separate, contiguous and 16-byte aligned. The context
    owns the registered scratch buffer; main() installs the TMA allocator.
    """
    _launch_reduce_scatter(input_tensor, ctx, config=config)
    ctx.calls += 1
    ctx.last_config = dict(config)
    return ctx.output


def torch_reduce_scatter(input_tensor, group):
    M, N = input_tensor.shape
    world_size = dist.get_world_size(group)
    output = torch.empty((M // world_size, N), dtype=input_tensor.dtype, device=input_tensor.device)
    dist.reduce_scatter_tensor(output, input_tensor, group=group)
    return output


def _assert_close_on_all_ranks(actual, expected, *, shape, stage, atol=6e-2, rtol=6e-2):
    """Raise on all ranks if any result fails validation."""
    local_correct = torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
    correct = torch.tensor(int(local_correct), dtype=torch.int32, device=actual.device)
    dist.all_reduce(correct, op=dist.ReduceOp.MIN)
    if correct.item() == 0:
        if not local_correct:
            max_error = (actual.float() - expected.float()).abs().max().item()
            print(f"[Rank {dist.get_rank()}] shape={shape} stage={stage} "
                  f"FAILED max_abs_error={max_error}", flush=True)
        raise AssertionError(f"shape={shape} stage={stage}: distributed correctness failed")


def benchmark_tle_vs_torch(input_tensor, ctx, config, *, warmup=WARMUP_ITERS, iters=BENCH_ITERS, clear_l2=True):
    """Measure the selected operation and torch using max(per-rank median)."""
    if warmup < 0 or iters < 1:
        raise ValueError("warmup must be nonnegative and iters must be positive")
    driver = triton.runtime.driver.active
    cache = driver.get_empty_cache_for_benchmark() if clear_l2 else None
    torch_output = torch.empty_like(ctx.output)

    def run_tle():
        # Exclude autotune's host-side candidate/cache agreement from timing.
        reduce_scatter.fn(input_tensor, ctx, config=config)

    def run_torch():
        dist.reduce_scatter_tensor(torch_output, input_tensor, group=dist.group.WORLD)

    def time_one(fn):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        dist.barrier()
        if clear_l2:
            driver.clear_cache(cache)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    for i in range(warmup):
        dist.barrier()
        for fn in ((run_torch, run_tle) if i % 2 else (run_tle, run_torch)):
            fn()
    torch.cuda.synchronize()

    tle_times, torch_times = [], []
    for i in range(iters):
        # Alternate execution order to reduce measurement bias.
        order = ((run_tle, tle_times), (run_torch, torch_times))
        for fn, samples in (reversed(order) if i % 2 else order):
            samples.append(time_one(fn))
    medians = torch.tensor(
        [statistics.median(tle_times), statistics.median(torch_times)],
        dtype=torch.float64,
        device=input_tensor.device,
    )
    dist.all_reduce(medians, op=dist.ReduceOp.MAX)
    return medians.tolist()


def main():
    M, N, dtype = _get_problem()
    clear_l2_value = os.environ.get("TLE_CLEAR_L2", "1").strip().lower()
    if clear_l2_value not in ("1", "true", "yes", "on", "0", "false", "no", "off"):
        raise ValueError("TLE_CLEAR_L2 must be boolean (0 or 1)")
    clear_l2 = clear_l2_value in ("1", "true", "yes", "on")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    mem_pool = tle.get_mem_pool()
    if mem_pool is None:
        raise RuntimeError("FlagCX memory pool is unavailable; check FlagCX build and environment variables.")

    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    if world_size < 2 or world_size != local_world_size or rank != local_rank:
        raise ValueError("This intra-node test requires one node with at least two GPUs and rank == LOCAL_RANK")
    group = dist.group.WORLD
    if torch.cuda.get_device_capability(local_rank)[0] < 9:
        raise ValueError("The TMA reduction requires sm90 or newer")
    if M % world_size:
        raise ValueError(f"M={M} must be divisible by world_size={world_size}")

    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    M_per_rank = M // world_size
    with torch.cuda.use_mem_pool(mem_pool):
        scatter_buf = torch.empty((M, N), dtype=dtype, device="cuda")
    ctx = ReduceScatterContext(
        scatter_buf=scatter_buf,
        scatter_ctx=tle.create_dist_tensor(scatter_buf),
        output=torch.empty((M_per_rank, N), dtype=dtype, device="cuda"),
        M_per_rank=M_per_rank,
        N=N,
        local_rank=local_rank,
        world_size=world_size,
        num_sms=torch.cuda.get_device_properties(local_rank).multi_processor_count,
        max_shared_memory=triton.runtime.driver.active.utils.get_device_properties(local_rank)["max_shared_mem"],
    )
    input_tensor = torch.empty((M, N), dtype=dtype, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(INPUT_SEED + rank)
    input_tensor.uniform_(-1, 1, generator=generator).mul_((rank + 1) / world_size)

    if rank == 0:
        print(
            f"shape={(M, N)} dtype={dtype} world_size={world_size}; "
            f"{len(CONFIG_SPACE)} raw candidates, filtered by prune_fn", flush=True)

    # The decorator handles pruning, rank intersection, tuning, persistence
    # and the final execution. Changed values must reuse the cached config.
    for iteration in range(2):
        if iteration:
            input_tensor.mul_(0.5)
        expected = torch_reduce_scatter(input_tensor, group)
        ctx.scatter_buf.fill_(float("nan"))
        ctx.output.fill_(float("nan"))
        previous_calls = ctx.calls
        actual = reduce_scatter(input_tensor, ctx, autotune_pg=group)
        _assert_close_on_all_ranks(actual, expected, shape=(M, N), stage=f"decorated_call_{iteration + 1}")
        calls = torch.tensor(ctx.calls - previous_calls, dtype=torch.int32, device="cuda")
        dist.all_reduce(calls, op=dist.ReduceOp.MAX, group=group)
        if iteration and calls.item() != 1:
            raise AssertionError("The second decorated call should execute the cached winner exactly once")
        if rank == 0:
            print(f"Pass {iteration + 1}: correct; operator invocations={calls.item()}; "
                  f"selected={ctx.last_config}", flush=True)

    selected = dict(ctx.last_config)
    # Validate exact integer sums with changed inputs and local consumers,
    # without host collectives between consecutive launches of the winner.
    rows = torch.arange(M, device="cuda", dtype=torch.int32)[:, None]
    cols = torch.arange(N, device="cuda", dtype=torch.int32)[None, :]
    exact_input = ((rows + 3 * cols + rank) % 5 - 2).to(dtype)
    exact_expected = exact_input.float()
    dist.all_reduce(exact_expected)
    exact_expected = exact_expected[rank * M_per_rank:(rank + 1) * M_per_rank].contiguous()
    snapshots = [torch.empty_like(ctx.output) for _ in range(2)]
    for snapshot in snapshots:
        exact_input.neg_()
        reduce_scatter.fn(exact_input, ctx, config=selected)
        snapshot.copy_(ctx.output)
    for iteration, snapshot in enumerate(snapshots):
        integer_expected = -exact_expected if iteration == 0 else exact_expected
        _assert_close_on_all_ranks(
            snapshot,
            integer_expected,
            shape=(M, N),
            stage=f"back_to_back_{iteration + 1}",
            atol=0.0,
            rtol=0.0,
        )

    tle_ms, torch_ms = benchmark_tle_vs_torch(input_tensor, ctx, selected, clear_l2=clear_l2)
    _assert_close_on_all_ranks(ctx.output, expected, shape=(M, N), stage="post_benchmark")
    if rank == 0:
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        print(f"Benchmark: iters={BENCH_ITERS} warmup={WARMUP_ITERS} clear_l2={clear_l2}; "
              "stat=max(per-rank median)", flush=True)
        print(
            f"TLE: {tle_ms:.3f} ms, {input_bytes / (tle_ms * 1e6):.2f} GB/s; "
            f"Torch: {torch_ms:.3f} ms, {input_bytes / (torch_ms * 1e6):.2f} GB/s; "
            f"speedup={torch_ms / tle_ms:.2f}x", flush=True)

    torch.cuda.synchronize()
    dist.barrier()
    tle.cleanup_communicator()


if __name__ == "__main__":
    main()
