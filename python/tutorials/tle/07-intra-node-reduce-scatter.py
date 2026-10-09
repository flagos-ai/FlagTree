"""Intra-node reduce-scatter with TMA reduction.

Run from the repository root on one node with at least two GPUs:

    NPROC_PER_NODE=4 M=2048 N=2048 DTYPE=bf16 \
        bash python/tutorials/tle/06-intra-node-reduce-scatter.sh

MASTER_ADDR defaults to localhost; MASTER_PORT defaults to 29501.

M/N default to 2048/2048; DTYPE accepts bf16 or fp16 (default: bf16).
NPROC_PER_NODE defaults to 4; use CUDA_VISIBLE_DEVICES to select GPUs.
Requires SM90 or newer and matching FlagTree/FlagCX runtime and device bitcode.
Set FLAGCX_IB_HCA to the interfaces on your machine when needed.
"""

import json
import os
import random
import statistics
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

import triton
import triton.language as tl
import triton.runtime
import triton.experimental.tle.language as tle


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
        index=0,
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


def _kernel_config(
    scatter_tile,
    reduce_tile,
    scatter_warps=4,
    reduce_warps=4,
    scatter_grid_factor=2,
):
    return {
        "SCATTER_BLOCK_M": scatter_tile[0],
        "SCATTER_BLOCK_N": scatter_tile[1],
        "REDUCE_BLOCK_M": reduce_tile[0],
        "REDUCE_BLOCK_N": reduce_tile[1],
        "scatter_num_warps": scatter_warps,
        "reduce_num_warps": reduce_warps,
        "scatter_grid_factor": scatter_grid_factor,
    }


# Sample joint tile/warp configurations without an exhaustive Cartesian search.
_BASE_KERNEL_CONFIGS = [
    _kernel_config((64, 128), (64, 128)),
    _kernel_config((64, 256), (64, 256)),
    _kernel_config((64, 256), (128, 128)),
    _kernel_config((128, 64), (128, 64)),
    _kernel_config((128, 128), (128, 128)),
    _kernel_config((128, 256), (128, 256)),
    _kernel_config((128, 256), (128, 256), reduce_warps=8),
    _kernel_config((64, 256), (256, 128)),
    _kernel_config((128, 256), (256, 128)),
    _kernel_config((128, 256), (256, 128), reduce_warps=8),
    _kernel_config((256, 128), (256, 64)),
    _kernel_config((256, 128), (256, 128)),
    _kernel_config((256, 128), (256, 128), scatter_warps=8),
]

# Tune scatter concurrency independently. Reduce stays capped at 1x num_sms.
SCATTER_GRID_FACTORS = (1, 2, 4)
KERNEL_CONFIGS = [{**config, "scatter_grid_factor": grid_factor}
                  for config in _BASE_KERNEL_CONFIGS
                  for grid_factor in SCATTER_GRID_FACTORS]


def _grid_ctas(M_per_rank, N, block_m, block_n, num_sms, grid_factor=1):
    """Cap the grid at grid_factor * num_sms; -1 uses the full tile grid."""
    if num_sms != -1 and num_sms < 1:
        raise ValueError("num_sms must be positive or -1 for an uncapped grid")
    if grid_factor < 1:
        raise ValueError("grid_factor must be positive")
    tiles = triton.cdiv(M_per_rank, block_m) * triton.cdiv(N, block_n)
    if num_sms == -1:
        return tiles
    return min(tiles, grid_factor * num_sms)


def _valid_configs(M_per_rank, N, dtype_nbytes, max_shared_memory):
    """Filter configurations by alignment and shared-memory requirements."""
    valid = []
    for config in KERNEL_CONFIGS:
        sm, sn = config["SCATTER_BLOCK_M"], config["SCATTER_BLOCK_N"]
        rm, rn = config["REDUCE_BLOCK_M"], config["REDUCE_BLOCK_N"]
        if sm > M_per_rank or sn > N or rm > M_per_rank or rn > N:
            continue
        # TMA loads must not cross a source-rank row boundary.
        if M_per_rank % rm:
            continue
        # Estimate the two reduction tiles and barrier state in shared memory.
        if 2 * rm * rn * dtype_nbytes + 8 > max_shared_memory:
            continue
        valid.append(config)
    return valid


AUTOTUNE_WARMUP = 10
AUTOTUNE_ITERS = 50
AUTOTUNE_TOP_K = 5
AUTOTUNE_RECHECK_ITERS = 100
AUTOTUNE_RECHECK_ROUNDS = 2
AUTOTUNE_SEED = 0
WARMUP_ITERS = 10
BENCH_ITERS = 200


def _save_autotune_result(
    *,
    M: int,
    N: int,
    world_size: int,
    local_world_size: int,
    dtype: torch.dtype,
    num_sms: int,
    best_config: dict,
    best_results: dict,
    tle_stats: dict,
    torch_stats: dict,
    speedup: float,
    valid_config_count: int,
    tuning_summary: dict,
    hardware: list,
    rank: int,
) -> None:
    """Save tuning results and measurements for the current shape."""
    if rank != 0:
        return

    output_value = os.environ.get(
        "TLE_AUTOTUNE_OUTPUT",
        "tle_intra_node_rs_autotune_results.json",
    ).strip()
    if not output_value:
        return

    output_path = Path(output_value)
    if output_path.exists():
        data = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        data = {"version": 1, "results": {}}
    if not isinstance(data, dict) or not isinstance(data.get("results"), dict):
        raise ValueError(f"invalid autotune result file: {output_path}")

    dtype_name = str(dtype).removeprefix("torch.")
    hardware_key = json.dumps(hardware, sort_keys=True, separators=(",", ":"))
    key = (f"op=intra_node_reduce_scatter/hardware={hardware_key}/shape={M}x{N}/"
           f"world={world_size}/local_world={local_world_size}/dtype={dtype_name}/"
           f"clear_l2={int(best_results['clear_l2'])}/stat=max_rank_median")
    data["results"][key] = {
        "op": "intra_node_reduce_scatter",
        "shape": [M, N],
        "output_shape": [M // world_size, N],
        "world_size": world_size,
        "local_world_size": local_world_size,
        "dtype": dtype_name,
        "num_sms": num_sms,
        "hardware": hardware,
        "torch_version": str(torch.__version__),
        "triton_version": triton.__version__,
        "best_config": dict(best_config),
        "autotune": {
            "candidate_count": valid_config_count,
            "warmup": AUTOTUNE_WARMUP,
            "iters": AUTOTUNE_ITERS,
            "seed": AUTOTUNE_SEED,
            "top_k": AUTOTUNE_TOP_K,
            "recheck_iters": AUTOTUNE_RECHECK_ITERS,
            "recheck_rounds": AUTOTUNE_RECHECK_ROUNDS,
            **tuning_summary,
        },
        "benchmark": {
            "warmup": WARMUP_ITERS,
            "iters": BENCH_ITERS,
            "input_bytes": best_results["input_bytes"],
            "clear_l2": best_results["clear_l2"],
            "latency_statistic": best_results["latency_statistic"],
            "tle": {
                "median_ms": tle_stats["median_ms"],
                "median_gbps": tle_stats["median_gbps"],
                "rank0_all_times_ms": tle_stats["local_all_times_ms"],
            },
            "torch": {
                "median_ms": torch_stats["median_ms"],
                "median_gbps": torch_stats["median_gbps"],
                "rank0_all_times_ms": torch_stats["local_all_times_ms"],
            },
            "speedup": speedup,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)
    print(f"Saved autotune result to {output_path}", flush=True)


def torch_reduce_scatter(input_tensor, group):
    M, N = input_tensor.shape
    world_size = dist.get_world_size(group)
    output = torch.empty((M // world_size, N), dtype=input_tensor.dtype, device=input_tensor.device)
    dist.reduce_scatter_tensor(output, input_tensor, group=group)
    return output


# Scatter -> device barrier -> TMA reduction.
def tle_reduce_scatter_device_barrier(
    input_tensor,
    scatter_buf,
    scatter_ctx,
    output,
    M_per_rank,
    N,
    local_rank,
    world_size,
    stream,
    num_sms: int = -1,
    config: Optional[dict] = None,
    compile_only: bool = False,
):
    if config is None:
        config = _BASE_KERNEL_CONFIGS[0]

    SCATTER_BLOCK_M = config["SCATTER_BLOCK_M"]
    SCATTER_BLOCK_N = config["SCATTER_BLOCK_N"]
    REDUCE_BLOCK_M = config["REDUCE_BLOCK_M"]
    REDUCE_BLOCK_N = config["REDUCE_BLOCK_N"]
    scatter_num_warps = config["scatter_num_warps"]
    reduce_num_warps = config["reduce_num_warps"]
    scatter_grid_factor = config["scatter_grid_factor"]

    grid_scatter = (_grid_ctas(
        M_per_rank,
        N,
        SCATTER_BLOCK_M,
        SCATTER_BLOCK_N,
        num_sms,
        grid_factor=scatter_grid_factor,
    ), )
    with torch.cuda.stream(stream):
        scatter_kernel = scatter_kernel_opt.run(
            input_tensor,
            scatter_buf,
            grid=grid_scatter,
            warmup=compile_only,
            scatter_ctx=scatter_ctx,
            M_per_rank=M_per_rank,
            N=N,
            LOCAL_RANK=local_rank,
            WORLD_SIZE=world_size,
            SCATTER_BLOCK_M=SCATTER_BLOCK_M,
            SCATTER_BLOCK_N=SCATTER_BLOCK_N,
            num_warps=scatter_num_warps,
        )

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid_reduce = (_grid_ctas(
        M_per_rank,
        N,
        REDUCE_BLOCK_M,
        REDUCE_BLOCK_N,
        num_sms,
        grid_factor=1,
    ), )

    with torch.cuda.stream(stream):

        barrier_kernel = device_barrier_kernel.run(
            grid=(1, ),
            warmup=compile_only,
            scatter_ctx=scatter_ctx,
            mesh=tle.device_mesh(tle.MeshConfig(device=world_size)),
        )
        reduce_kernel = ring_reduce_kernel_tma.run(
            scatter_buf,
            output,
            M_per_rank,
            N,
            grid=grid_reduce,
            warmup=compile_only,
            LOCAL_RANK=local_rank,
            WORLD_SIZE=world_size,
            REDUCE_BLOCK_M=REDUCE_BLOCK_M,
            REDUCE_BLOCK_N=REDUCE_BLOCK_N,
            num_warps=reduce_num_warps,
        )

    if compile_only:
        # Loading binaries checks actual resources without executing any kernel.
        for kernel in (scatter_kernel, barrier_kernel, reduce_kernel):
            if hasattr(kernel, "result"):
                kernel = kernel.result()
            kernel._init_handles()


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
        dist.barrier()
        raise AssertionError(f"shape={shape} stage={stage}: distributed correctness failed")


def benchmark_tle_vs_torch(
    input_tensor,
    scatter_buf,
    scatter_ctx,
    torch_output_bench,
    M_per_rank,
    N,
    local_rank,
    world_size,
    stream,
    num_sms: int = -1,
    config: Optional[dict] = None,
    warmup: int = WARMUP_ITERS,
    iters: int = BENCH_ITERS,
    benchmark_torch: bool = True,
    clear_l2: bool = True,
):
    """Return the maximum of per-rank median latencies.

    Each invocation is synchronized; clear_l2 optionally clears the cache.
    """
    if warmup < 0 or iters < 1:
        raise ValueError("warmup must be nonnegative and iters must be positive")
    elem_size = input_tensor.element_size()
    input_bytes = input_tensor.numel() * elem_size

    driver = triton.runtime.driver.active
    cache = driver.get_empty_cache_for_benchmark() if clear_l2 else None

    def run_tle():
        tle_reduce_scatter_device_barrier(
            input_tensor,
            scatter_buf,
            scatter_ctx,
            torch_output_bench,
            M_per_rank,
            N,
            local_rank,
            world_size,
            stream,
            num_sms=num_sms,
            config=config,
        )

    def run_torch():
        with torch.cuda.stream(stream):
            dist.reduce_scatter_tensor(torch_output_bench, input_tensor, group=None)

    def _time_one(fn):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        dist.barrier()
        with torch.cuda.stream(stream):
            if clear_l2:
                driver.clear_cache(cache)
            start.record(stream)
            fn()
            end.record(stream)
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    for i in range(warmup):
        dist.barrier()
        if benchmark_torch and i % 2:
            run_torch()
            run_tle()
        else:
            run_tle()
            if benchmark_torch:
                run_torch()
    torch.cuda.synchronize()

    tle_times = []
    torch_times = []
    for i in range(iters):
        # Alternate execution order to reduce measurement bias.
        if benchmark_torch and i % 2:
            torch_times.append(_time_one(run_torch))
            tle_times.append(_time_one(run_tle))
        else:
            tle_times.append(_time_one(run_tle))
            if benchmark_torch:
                torch_times.append(_time_one(run_torch))

    # Take each rank's median first, then the maximum across ranks.
    # Keep raw samples local; aggregation stays outside the timed regions.
    local_medians = [float(statistics.median(tle_times))]
    if benchmark_torch:
        local_medians.append(float(statistics.median(torch_times)))
    medians = torch.tensor(
        local_medians,
        dtype=torch.float64,
        device=input_tensor.device,
    )
    dist.all_reduce(medians, op=dist.ReduceOp.MAX)
    global_medians = medians.tolist()

    def _gbps(bytes_moved, ms):
        return (bytes_moved / (ms * 1e-3)) / 1e9

    tle_median_ms = global_medians[0]
    torch_median_ms = global_medians[1] if benchmark_torch else None

    return {
        "tle": {
            "median_ms": tle_median_ms,
            "median_gbps": _gbps(input_bytes, tle_median_ms),
            "local_all_times_ms": tle_times,
        },
        "torch": None if not benchmark_torch else {
            "median_ms": torch_median_ms,
            "median_gbps": _gbps(input_bytes, torch_median_ms),
            "local_all_times_ms": torch_times,
        },
        "input_bytes": input_bytes,
        "world_size": world_size,
        "clear_l2": clear_l2,
        "latency_statistic": "max_of_per_rank_median",
    }


def _select_best_config(configs, benchmark, *, warmup, iters):
    """Screen candidates, then remeasure finalists in alternating order.

    benchmark returns globally aggregated scores, or None when all ranks
    agreed to skip a candidate during compilation preflight.
    """
    if not configs:
        raise ValueError("autotuning requires at least one candidate")
    order = list(range(len(configs)))
    random.Random(AUTOTUNE_SEED).shuffle(order)
    screening = []
    skipped = []
    for index in order:
        result = benchmark(configs[index], warmup, iters)
        if result is None:
            skipped.append(index)
            continue
        screening.append((result["tle"]["median_ms"], index))
    if not screening:
        raise RuntimeError("all autotuning candidates failed compilation/resource checks")

    finalists = [index for _, index in sorted(screening)[:AUTOTUNE_TOP_K]]
    repeated_scores = {index: [] for index in finalists}
    for round_index in range(AUTOTUNE_RECHECK_ROUNDS):
        if dist.get_rank() == 0:
            print(
                f"Recheck round {round_index + 1}/{AUTOTUNE_RECHECK_ROUNDS}: "
                f"{len(finalists)} finalists, {AUTOTUNE_RECHECK_ITERS} iterations", flush=True)
        round_order = finalists if round_index % 2 == 0 else list(reversed(finalists))
        for index in round_order:
            result = benchmark(configs[index], warmup, AUTOTUNE_RECHECK_ITERS)
            if result is None:
                raise RuntimeError("a previously compiled finalist became unavailable")
            repeated_scores[index].append(result["tle"]["median_ms"])
    # Each round uses max(per-rank median); summarize independent rechecks.
    best_index = min(finalists, key=lambda index: (statistics.median(repeated_scores[index]), index))
    summary = {
        "skipped_configs": [configs[index] for index in skipped],
        "screening": [{"config": configs[index], "median_ms": score} for score, index in screening],
        "finalists": [{
            "config": configs[index], "round_scores_ms": repeated_scores[index], "selection_score_ms":
            statistics.median(repeated_scores[index])
        } for index in finalists],
    }
    return configs[best_index], summary


def main():
    M, N, dtype = _get_problem()
    clear_l2_value = os.environ.get("TLE_CLEAR_L2", "1").strip().lower()
    if clear_l2_value not in ("1", "true", "yes", "on", "0", "false", "no", "off"):
        raise ValueError("TLE_CLEAR_L2 must be boolean (0 or 1)")
    clear_l2 = clear_l2_value in ("1", "true", "yes", "on")
    mem_pool = tle.get_mem_pool()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    torch.cuda.set_device(local_rank)

    print(f"[Rank {rank}/{world_size}] Starting TLE reduce-scatter (TMA reduce, optimized scatter, device barrier)")

    if world_size < 2 or world_size != local_world_size:
        raise ValueError("This intra-node test requires one node with at least two GPUs")
    if torch.cuda.get_device_capability(local_rank)[0] < 9:
        raise ValueError("The TMA reduction requires sm90 or newer")
    if M % world_size:
        raise ValueError(f"M={M} must be divisible by world_size={world_size}")
    dtype_nbytes = torch.empty((), dtype=dtype).element_size()
    if (N * dtype_nbytes) % 16:
        raise ValueError(f"N={N} must give a 16-byte-aligned row stride for {dtype}")

    device_props = torch.cuda.get_device_properties(local_rank)
    num_sms = device_props.multi_processor_count
    max_shared_memory = triton.runtime.driver.active.utils.get_device_properties(local_rank)["max_shared_mem"]
    hardware = [None] * world_size
    dist.all_gather_object(
        hardware, {
            "name": device_props.name,
            "capability": list(torch.cuda.get_device_capability(local_rank)),
            "num_sms": num_sms,
            "total_memory": device_props.total_memory,
        })
    # Every rank must enumerate the same candidate list.
    shared_limit = torch.tensor(max_shared_memory, dtype=torch.int64, device="cuda")
    dist.all_reduce(shared_limit, op=dist.ReduceOp.MIN)
    max_shared_memory = int(shared_limit.item())
    M_per_rank = M // world_size
    valid_configs = _valid_configs(M_per_rank, N, dtype_nbytes, max_shared_memory)
    if not valid_configs:
        raise ValueError(f"no config fits shape {(M, N)}, dtype={dtype}, world_size={world_size}")
    if rank == 0:
        print(
            f"shape={(M, N)} dtype={dtype}: {len(valid_configs)}/{len(KERNEL_CONFIGS)} "
            "candidates after shape/resource filtering", flush=True)

    with torch.cuda.use_mem_pool(mem_pool):
        scatter_buf = torch.empty((M, N), dtype=dtype, device="cuda")
    scatter_ctx = tle.create_dist_tensor(scatter_buf)

    # Exercise cancellation with distinct inputs across ranks.
    input_tensor = torch.empty((M, N), dtype=dtype, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(AUTOTUNE_SEED + rank)
    input_tensor.uniform_(-1, 1, generator=generator).mul_((rank + 1) / world_size)

    output = torch.empty((M_per_rank, N), dtype=dtype, device="cuda")

    stream = torch.cuda.current_stream()

    torch_output = torch_reduce_scatter(input_tensor, group=None)
    torch.cuda.synchronize()

    # Small integer sums are exact for the supported low-precision dtypes
    # at normal intra-node rank counts. Different ranks and rows carry patterns.
    rows = torch.arange(M, device="cuda", dtype=torch.int32)[:, None]
    cols = torch.arange(N, device="cuda", dtype=torch.int32)[None, :]
    exact_input = ((rows + 3 * cols + rank) % 5 - 2).to(dtype)
    exact_expected = exact_input.float()
    dist.all_reduce(exact_expected)
    exact_expected = exact_expected[rank * M_per_rank:(rank + 1) * M_per_rank].contiguous()
    del rows, cols

    def check_config(cfg, stage):
        for label, check_input, expected, atol, rtol in (
            ("integer", exact_input, exact_expected, 0.0, 0.0),
            ("random", input_tensor, torch_output, 6e-2, 6e-2),
        ):
            # Poison both buffers. Finish poisoning on ALL ranks before remote writes.
            scatter_buf.fill_(float("nan"))
            output.fill_(float("nan"))
            torch.cuda.synchronize()
            dist.barrier()
            tle_reduce_scatter_device_barrier(
                check_input,
                scatter_buf,
                scatter_ctx,
                output,
                M_per_rank,
                N,
                local_rank,
                world_size,
                stream,
                num_sms=num_sms,
                config=cfg,
            )
            torch.cuda.synchronize()
            _assert_close_on_all_ranks(
                output,
                expected,
                shape=(M, N),
                stage=f"{stage}/{label}",
                atol=atol,
                rtol=rtol,
            )

    compiled_configs = set()

    def preflight_config(cfg):
        index = valid_configs.index(cfg)
        if index in compiled_configs:
            return True
        error = None
        try:
            tle_reduce_scatter_device_barrier(
                input_tensor,
                scatter_buf,
                scatter_ctx,
                output,
                M_per_rank,
                N,
                local_rank,
                world_size,
                stream,
                num_sms=num_sms,
                config=cfg,
                compile_only=True,
            )
        except Exception as exc:
            # Only compilation/loading is caught; execution/correctness failures remain fatal.
            error = f"{type(exc).__name__}: {exc}"
        ok = torch.tensor(int(error is None), dtype=torch.int32, device="cuda")
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        if not ok.item():
            if error is not None:
                print(f"[Rank {rank}] Candidate {index + 1} preflight failed: {error}", flush=True)
            if rank == 0:
                print(f"Skipping candidate {index + 1} on all ranks", flush=True)
            return False
        compiled_configs.add(index)
        return True

    torch_output_bench = torch.empty_like(output)

    def _benchmark_with_config(cfg, warmup, iters):
        if rank == 0:
            print(f"Candidate {valid_configs.index(cfg) + 1}/{len(valid_configs)}: {cfg}", flush=True)
        if not preflight_config(cfg):
            return None
        check_config(cfg, stage=f"candidate {valid_configs.index(cfg) + 1}/{len(valid_configs)}")
        return benchmark_tle_vs_torch(
            input_tensor,
            scatter_buf,
            scatter_ctx,
            torch_output_bench,
            M_per_rank,
            N,
            local_rank,
            world_size,
            stream,
            num_sms=num_sms,
            config=cfg,
            warmup=warmup,
            iters=iters,
            benchmark_torch=False,
            clear_l2=clear_l2,
        )

    best_cfg, tuning_summary = _select_best_config(
        valid_configs,
        _benchmark_with_config,
        warmup=AUTOTUNE_WARMUP,
        iters=AUTOTUNE_ITERS,
    )

    check_config(best_cfg, stage="selected")

    best_results = benchmark_tle_vs_torch(
        input_tensor,
        scatter_buf,
        scatter_ctx,
        torch_output_bench,
        M_per_rank,
        N,
        local_rank,
        world_size,
        stream,
        num_sms=num_sms,
        config=best_cfg,
        warmup=WARMUP_ITERS,
        iters=BENCH_ITERS,
        benchmark_torch=True,
        clear_l2=clear_l2,
    )

    # Rerun TLE because benchmarking shares the output buffer with PyTorch.
    check_config(best_cfg, stage="post_benchmark")
    tle_stats = best_results["tle"]
    torch_stats = best_results["torch"]

    log = print if rank == 0 else lambda *args, **kwargs: None
    log(f"[Rank {rank}] shape={input_tensor.shape} dtype={input_tensor.dtype} "
        f"world_size={world_size} iters={BENCH_ITERS} (warmup={WARMUP_ITERS}) "
        f"num_sms={num_sms} clear_l2={clear_l2} "
        "stat=max(per-rank median)")
    log(f"[Rank {rank}] best_config={best_cfg} "
        f"TLE    median={tle_stats['median_ms']:.3f}ms "
        f"alg_bw={tle_stats['median_gbps']:.2f}GB/s")
    log(f"[Rank {rank}] Torch  median={torch_stats['median_ms']:.3f}ms "
        f"alg_bw={torch_stats['median_gbps']:.2f}GB/s")
    speedup = torch_stats["median_ms"] / tle_stats["median_ms"]
    log(f"[Rank {rank}] TLE/Torch = {speedup:.2f}x")

    _save_autotune_result(
        M=M,
        N=N,
        world_size=world_size,
        local_world_size=local_world_size,
        dtype=dtype,
        num_sms=num_sms,
        best_config=best_cfg,
        best_results=best_results,
        tle_stats=tle_stats,
        torch_stats=torch_stats,
        speedup=speedup,
        valid_config_count=len(valid_configs),
        tuning_summary=tuning_summary,
        hardware=hardware,
        rank=rank,
    )

    torch.cuda.synchronize()
    dist.barrier()

    tle.cleanup_communicator()


if __name__ == "__main__":
    main()
