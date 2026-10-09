"""Persistent GEMM with hierarchical reduce-scatter.

Run once on each node from the repository root:

    # Node 0
    NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 NPROC_PER_NODE=4 \
        bash python/tutorials/tle/08-inter-node-gemm-reduce-scatter.sh --M 2048 --N 4096 --K 16384 --dtype bf16
    # Node 1
    NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 NPROC_PER_NODE=4 \
        bash python/tutorials/tle/08-inter-node-gemm-reduce-scatter.sh --M 2048 --N 4096 --K 16384 --dtype bf16

Use matching topology, dimensions, dtype, and tuning settings on every node.
MASTER_ADDR must be reachable from all nodes; MASTER_PORT defaults to 29501.

--M/--N/--K default to 2048/4096/16384. K is global: each rank uses K/world_size.
--dtype accepts bf16 or fp16 and overrides DTYPE (default: bf16).
NPROC_PER_NODE defaults to gpu (one process per visible GPU).
Set CUDA_VISIBLE_DEVICES externally to select GPUs.
FLAGCX_IB_HCA is configured in the launch script.
Requires SM90 or newer and matching FlagTree/FlagCX runtime and device bitcode.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import triton
import triton.runtime
import triton.language as tl
import triton.experimental.tle.language as tle

# Numbered tutorial filenames are loaded through importlib.
_rs = importlib.import_module("07-inter-node-reduce-scatter")
TleReduceScatter2DContext = _rs.TleReduceScatter2DContext
create_tle_reduce_scatter_2d_ctx = _rs.create_tle_reduce_scatter_2d_ctx
reduce_scatter_2d_op = _rs.reduce_scatter_2d_op
validate_node_transfer_shape = _rs.validate_node_transfer_shape
get_test_dtype = _rs.get_test_dtype
_assert_close_on_all_ranks = _rs._assert_close_on_all_ranks
RS_KERNEL_CONFIGS = _rs.KERNEL_CONFIGS


@dataclasses.dataclass
class TleGemmConfig:
    """Persistent GEMM launch configuration."""

    block_m: int = 128
    block_n: int = 256
    block_k: int = 64
    group_m: int = 8
    num_warps: int = 8
    num_stages: int = 4
    persistent: bool = True
    fuse_scatter: bool = False
    # Split the output tile into two stores along N.
    epilogue_subtile: bool = False

    def validate(self) -> None:
        for name in ("block_m", "block_n", "block_k", "group_m", "num_warps", "num_stages"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not self.persistent:
            raise ValueError("multi-node GEMM+RS requires persistent=True")
        if self.fuse_scatter:
            raise ValueError("multi-node GEMM+RS requires fuse_scatter=False")
        if self.epilogue_subtile and self.block_n % 2:
            raise ValueError("BLOCK_SIZE_N must be even when EPILOGUE_SUBTILE=True")
        if self.num_warps not in (1, 2, 4, 8, 16, 32):
            raise ValueError("num_warps must be one of 1, 2, 4, 8, 16, 32")


@dataclasses.dataclass
class TleGemmReduceScatterContext:

    rs_ctx: TleReduceScatter2DContext
    output_dtype: torch.dtype
    rs_stream: torch.cuda.Stream
    num_gemm_sms: int
    gemm_config: TleGemmConfig

    def __post_init__(self):
        if self.rs_stream is self.rs_ctx.reduction_stream:
            raise ValueError("rs_stream and reduction_stream must be distinct")

    def finalize(self):
        self.rs_ctx.finalize()

    def get_gemm_out_buf(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.rs_ctx.gemm_out_buf is None:
            raise RuntimeError("GEMM context must reserve a symmetric GEMM output buffer")
        return self.rs_ctx.gemm_out_buf[:input_tensor.shape[0]]


def create_gemm_rs_context(max_M: int, N: int, rank: int, world_size: int, local_world_size: int,
                           output_dtype: torch.dtype, rs_stream: torch.cuda.Stream,
                           gemm_config: Optional[TleGemmConfig] = None, num_reduction_sms: int = 15,
                           num_scatter_sms: int = 16) -> TleGemmReduceScatterContext:
    """Create the persistent GEMM and reduce-scatter context."""

    if max_M % world_size:
        raise ValueError("max_M must be divisible by world_size")

    if gemm_config is None:
        gemm_config = TleGemmConfig()
    gemm_config.validate()
    rs_ctx = create_tle_reduce_scatter_2d_ctx(max_M, N, rank, world_size, local_world_size, output_dtype,
                                              with_gemm_output=True, num_reduction_sms=num_reduction_sms,
                                              num_scatter_sms=num_scatter_sms)

    total_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_gemm_sms = total_sms - rs_ctx.num_rs_sms

    if num_gemm_sms < 1:
        raise ValueError("reduce-scatter SM reservation leaves no SM for GEMM")

    return TleGemmReduceScatterContext(rs_ctx=rs_ctx, output_dtype=output_dtype, rs_stream=rs_stream,
                                       num_gemm_sms=num_gemm_sms, gemm_config=gemm_config)


@triton.jit
def kernel_gemm_rs_producer_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    ready_ptr,
    counter_ptr,
    RANK: tl.constexpr,
    LOCAL_WORLD_SIZE: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Compute output tiles and publish per-rank readiness."""

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    node_id = RANK // LOCAL_WORLD_SIZE
    nnodes = WORLD_SIZE // LOCAL_WORLD_SIZE
    tiles_per_sm = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_sm += 1

    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, K], strides=[K, 1], block_shape=[BLOCK_N, BLOCK_K])
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[
            BLOCK_M,
            BLOCK_N // 2 if EPILOGUE_SUBTILE else BLOCK_N,
        ],
    )

    M_per_rank = M // WORLD_SIZE
    tiles_m_per_rank = M_per_rank // BLOCK_M
    tiles_per_group = GROUP_M * num_pid_n
    tile_id = start_pid - NUM_SMS
    k_tile = -1
    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_sm):

        k_tile = tl.where(k_tile == k_tiles - 1, 0, k_tile + 1)

        if k_tile == 0:
            tile_id += NUM_SMS
            group_id = tile_id // tiles_per_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            logical_pid_m = first_pid_m + tile_id % group_size_m
            pid_n = (tile_id % tiles_per_group) // group_size_m
            m_rank = logical_pid_m // tiles_m_per_rank
            pid_m_intra_rank = logical_pid_m - m_rank * tiles_m_per_rank
            m_node_id = m_rank // LOCAL_WORLD_SIZE
            m_local_rank = m_rank % LOCAL_WORLD_SIZE
            swizzle_m_node_id = (m_node_id + node_id + 1) % nnodes
            swizzle_m_local_rank = (m_local_rank + RANK + 1) % LOCAL_WORLD_SIZE
            swizzle_m_rank = (swizzle_m_node_id * LOCAL_WORLD_SIZE + swizzle_m_local_rank)
            pid_m = swizzle_m_rank * tiles_m_per_rank + pid_m_intra_rank
            offs_am = pid_m * BLOCK_M
            offs_bn = pid_n * BLOCK_N

        a = a_desc.load([offs_am, k_tile * BLOCK_K])
        b = b_desc.load([offs_bn, k_tile * BLOCK_K])
        accumulator = tl.dot(a, b.T, accumulator)

        if k_tile == k_tiles - 1:

            if EPILOGUE_SUBTILE:
                acc = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
                acc = tl.permute(acc, (0, 2, 1))
                acc0, acc1 = tl.split(acc)
                c0 = acc0.to(c_ptr.dtype.element_ty)
                c_desc.store([offs_am, offs_bn], c0)
                c1 = acc1.to(c_ptr.dtype.element_ty)
                c_desc.store([offs_am, offs_bn + BLOCK_N // 2], c1)
            else:
                c_desc.store(
                    [offs_am, offs_bn],
                    accumulator.to(c_ptr.dtype.element_ty),
                )

            counter_start = offs_am // M_per_rank
            counter_end = (offs_am + BLOCK_M - 1) // M_per_rank
            counter_end = min(counter_end, WORLD_SIZE - 1)

            for counter_id in range(counter_start, counter_end + 1):
                m_start = M_per_rank * counter_id
                m_end = M_per_rank * (counter_id + 1) - 1
                tiled_m_start = m_start // BLOCK_M
                tiled_m_end = m_end // BLOCK_M
                tiled_m_size = tiled_m_end - tiled_m_start + 1
                tiled_n = tl.cdiv(N, BLOCK_N)

                # Order tile-counter updates before publishing ready.
                # TMA store completion is a separate requirement.
                prior = tl.atomic_add(counter_ptr + counter_id, 1, sem="acq_rel", scope="gpu")

                if prior == tiled_m_size * tiled_n - 1:
                    # The last tile publishes ready; RS clears the flag after use.
                    tl.atomic_xchg(ready_ptr + counter_id, 1, sem="release", scope="gpu")

            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


def gemm_rs_producer_persistent(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, barrier: torch.Tensor,
                                workspace: torch.Tensor, world_size: int, local_world_size: int, rank: int,
                                num_gemm_sms: int, BLOCK_SIZE_M: int = 128, BLOCK_SIZE_N: int = 256,
                                BLOCK_SIZE_K: int = 64, GROUP_SIZE_M: int = 8, STAGES: int = 4, NUM_WARPS: int = 8,
                                EPILOGUE_SUBTILE: bool = False):
    """Launch the persistent GEMM producer."""

    if a.shape[1] != b.shape[1]:
        raise ValueError("incompatible GEMM dimensions")
    if a.dtype != b.dtype:
        raise ValueError("GEMM operands must have the same dtype")
    M, local_K = a.shape
    N = b.shape[0]
    M_per_rank = M // world_size

    if M_per_rank % BLOCK_SIZE_M:
        raise ValueError("M_per_rank must be aligned to BLOCK_SIZE_M")

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    grid = lambda META: (min(
        num_gemm_sms,
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    ), )

    return kernel_gemm_rs_producer_persistent[grid](
        a,
        b,
        c,
        M,
        N,
        local_K,
        barrier,
        workspace,
        RANK=rank,
        LOCAL_WORLD_SIZE=local_world_size,
        WORLD_SIZE=world_size,
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_N=BLOCK_SIZE_N,
        BLOCK_K=BLOCK_SIZE_K,
        GROUP_M=GROUP_SIZE_M,
        EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
        NUM_SMS=num_gemm_sms,
        num_warps=NUM_WARPS,
        num_stages=STAGES,
    )


def _pad_to_block_m(input_tensor: torch.Tensor, world_size: int, block_m: int) -> torch.Tensor:
    """Pad rows to align each rank shard with GEMM tiles."""
    M, K = input_tensor.shape
    M_per_rank = M // world_size
    padded_M_per_rank = triton.cdiv(M_per_rank, block_m) * block_m
    if padded_M_per_rank == M_per_rank:
        return input_tensor
    reshaped = input_tensor.reshape(world_size, M_per_rank, K)
    padded = torch.empty((world_size, padded_M_per_rank, K), dtype=input_tensor.dtype, device=input_tensor.device)
    padded[:, :M_per_rank].copy_(reshaped)
    return padded.reshape(-1, K)


def gemm_rs_multi_node_persistent_op(input_tensor: torch.Tensor, weight: torch.Tensor, ctx: TleGemmReduceScatterContext,
                                     rs_config: Optional[dict] = None) -> torch.Tensor:
    """Overlap persistent GEMM with hierarchical reduce-scatter."""

    if rs_config is None:
        rs_config = RS_KERNEL_CONFIGS[0]

    world_size = ctx.rs_ctx.world_size
    local_world_size = ctx.rs_ctx.local_world_size
    rs_stream = ctx.rs_stream
    original_M = input_tensor.shape[0]
    original_M_per_rank = original_M // world_size

    input_tensor = _pad_to_block_m(input_tensor, world_size, ctx.gemm_config.block_m)

    M, K = input_tensor.shape
    N = weight.shape[0]
    if N != ctx.rs_ctx.N or weight.shape[1] != K:
        raise ValueError("invalid GEMM dimensions for the reduce-scatter context")
    if M > ctx.rs_ctx.max_M:
        raise ValueError("padded M exceeds context capacity")

    current_stream = torch.cuda.current_stream()

    rs_stream.wait_stream(current_stream)

    output = torch.empty((M // world_size, N), dtype=ctx.output_dtype, device="cuda")

    # workspace[j] counts completed output tiles for destination rank j.
    workspace = torch.zeros((world_size, ), dtype=torch.int32, device=input_tensor.device)
    scatter_signal = ctx.rs_ctx.scatter_signal_buf
    gemm_out = ctx.get_gemm_out_buf(input_tensor)

    gemm_rs_producer_persistent(
        input_tensor,
        weight,
        gemm_out,
        scatter_signal,
        workspace,
        world_size,
        local_world_size,
        ctx.rs_ctx.rank,
        ctx.num_gemm_sms,
        BLOCK_SIZE_M=ctx.gemm_config.block_m,
        BLOCK_SIZE_N=ctx.gemm_config.block_n,
        BLOCK_SIZE_K=ctx.gemm_config.block_k,
        GROUP_SIZE_M=ctx.gemm_config.group_m,
        STAGES=ctx.gemm_config.num_stages,
        NUM_WARPS=ctx.gemm_config.num_warps,
        EPILOGUE_SUBTILE=ctx.gemm_config.epilogue_subtile,
    )

    with torch.cuda.stream(rs_stream):
        reduce_scatter_2d_op(gemm_out, ctx.rs_ctx, output=output, ready_flags=scatter_signal, config=rs_config)
    current_stream.wait_stream(rs_stream)
    return output[:original_M_per_rank]


def gemm_rs_multi_node(input_tensor: torch.Tensor, weight: torch.Tensor, ctx: TleGemmReduceScatterContext,
                       rs_config: Optional[dict] = None) -> torch.Tensor:

    return gemm_rs_multi_node_persistent_op(input_tensor, weight, ctx, rs_config=rs_config)


def torch_gemm_rs(input_tensor: torch.Tensor, weight: torch.Tensor, TP_GROUP) -> torch.Tensor:

    M, _ = input_tensor.shape
    N = weight.shape[0]
    gemm_out = torch.matmul(input_tensor, weight.T)
    output = torch.empty((M // TP_GROUP.size(), N), dtype=gemm_out.dtype, device=input_tensor.device)
    dist.reduce_scatter_tensor(output, gemm_out, group=TP_GROUP)
    return output


def _time_ms(fn, stream: torch.cuda.Stream, warmup: int = 20, iters: int = 200, clear_l2: bool = True,
             group=None) -> float:
    """Return the median of per-sample maximum latencies across ranks."""

    driver = triton.runtime.driver.active
    cache = driver.get_empty_cache_for_benchmark() if clear_l2 else None

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        dist.barrier(group=group)
        if clear_l2:
            driver.clear_cache(cache)

        start.record(stream)
        fn()
        end.record(stream)
        torch.cuda.synchronize()
        # Take the maximum across ranks for each sample before computing the median.
        elapsed = torch.tensor(start.elapsed_time(end), dtype=torch.float64, device="cuda")
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
        samples.append(float(elapsed.item()))
    return float(statistics.median(samples))


def _env_positive_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer, got {parsed}")
    return parsed


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be boolean, got {value!r}")


def _get_epilogue_subtile_setting() -> Optional[bool]:
    """Parse the epilogue setting; None enables automatic selection."""
    value = os.environ.get("TLE_EPILOGUE_SUBTILE", "auto").strip().lower()
    if value in ("", "auto"):
        return None
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError("TLE_EPILOGUE_SUBTILE must be auto or boolean, "
                     f"got {value!r}")


def _get_gemm_config() -> TleGemmConfig:
    """Parse and validate the base GEMM configuration."""
    epilogue_setting = _get_epilogue_subtile_setting()
    config = TleGemmConfig(
        block_m=_env_positive_int("TLE_BLOCK_SIZE_M", 128),
        block_n=_env_positive_int("TLE_BLOCK_SIZE_N", 256),
        block_k=_env_positive_int("TLE_BLOCK_SIZE_K", 64),
        group_m=_env_positive_int("TLE_GROUP_SIZE_M", 8),
        num_warps=_env_positive_int("TLE_NUM_WARPS", 8),
        num_stages=_env_positive_int("TLE_NUM_STAGES", 4),
        persistent=_env_bool("TLE_PERSISTENT", True),
        fuse_scatter=_env_bool("TLE_FUSE_SCATTER", False),
        # Auto mode tests both epilogue variants.
        epilogue_subtile=False if epilogue_setting is None else epilogue_setting,
    )
    config.validate()
    return config


def _validate_problem_size(M: int, N: int, K: int, world_size: int, block_m: int = 128, block_n: int = 256,
                           block_k: int = 64) -> None:
    """Validate distributed dimensions and tile alignment."""
    if min(M, N, K) <= 0:
        raise ValueError("M, N and K must be positive")
    if M % world_size:
        raise ValueError(f"M={M} must be divisible by world_size={world_size}")
    if K % world_size:
        raise ValueError(f"K={K} must be divisible by world_size={world_size}")

    M_per_rank = M // world_size
    local_K = K // world_size

    # Require aligned shards so padding fits the allocated context.
    if M_per_rank < 256:
        raise ValueError(f"M/world_size={M_per_rank} must be >= 256 for the TMA reduce kernel")
    if M_per_rank % block_m:
        raise ValueError(f"M/world_size={M_per_rank} must be divisible by BLOCK_M={block_m}")
    if N % block_n:
        raise ValueError(f"N={N} must be divisible by GEMM BLOCK_N={block_n}")
    if local_K % block_k:
        raise ValueError(f"K/world_size={local_K} must be divisible by BLOCK_K={block_k}")


def _configure_sms(ctx: TleGemmReduceScatterContext, num_scatter_sms: int, num_reduction_sms: int) -> None:
    """Apply scatter and reduction SM budgets."""
    # SM budgets limit CTA counts, not physical SM placement.
    # The last local reduction and final reduction use full grids.
    if num_scatter_sms < 1 or num_reduction_sms < 1:
        raise ValueError("SMS budgets must be positive")

    ctx.rs_ctx.num_scatter_sms = num_scatter_sms
    ctx.rs_ctx.num_reduction_sms = num_reduction_sms
    total_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    ctx.num_gemm_sms = total_sms - ctx.rs_ctx.num_rs_sms
    if ctx.num_gemm_sms < 1:
        raise ValueError(f"SMS reservation leaves no SM for GEMM: "
                         f"scatter={num_scatter_sms}, reduction={num_reduction_sms}, "
                         f"total={total_sms}")


_GEMM_AUTOTUNE_CORE_CONFIGS = [
    # BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages
    (128, 256, 64, 8, 8, 4),
    (128, 256, 64, 8, 8, 3),
    (128, 256, 64, 8, 8, 5),
    (128, 128, 64, 8, 8, 4),
    (128, 128, 64, 8, 8, 3),
    (128, 128, 64, 4, 4, 4),
    (64, 256, 64, 8, 4, 4),
    (64, 256, 64, 8, 8, 4),
    (64, 128, 64, 8, 4, 4),
    (64, 128, 64, 8, 8, 4),
    (128, 256, 128, 8, 8, 3),
    (128, 256, 128, 8, 8, 4),
    (128, 128, 128, 8, 8, 3),
    (128, 128, 128, 8, 8, 4),
    (64, 256, 128, 8, 4, 4),
]

_GEMM_AUTOTUNE_CONFIGS = [(*core_config, epilogue_subtile)
                          for core_config in _GEMM_AUTOTUNE_CORE_CONFIGS
                          for epilogue_subtile in (False, True)]


def _gemm_config_key(config: TleGemmConfig) -> tuple:
    return (
        config.block_m,
        config.block_n,
        config.block_k,
        config.group_m,
        config.num_warps,
        config.num_stages,
        config.epilogue_subtile,
    )


def _gemm_config_dict(config: TleGemmConfig) -> dict:
    return {
        "BLOCK_M": config.block_m,
        "BLOCK_N": config.block_n,
        "BLOCK_K": config.block_k,
        "GROUP_M": config.group_m,
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
        "epilogue_subtile": config.epilogue_subtile,
        "persistent": config.persistent,
        "fuse_scatter": config.fuse_scatter,
    }


def _estimated_gemm_shared_memory_bytes(
    config: TleGemmConfig,
    element_size: int = 2,
) -> int:
    """Estimate staged inputs, output tiles, and shared-memory overhead."""
    staged_inputs = (config.num_stages * (config.block_m * config.block_k + config.block_n * config.block_k) *
                     element_size)
    output_block_n = (config.block_n // 2 if config.epilogue_subtile else config.block_n)
    output_tile = config.block_m * output_block_n * element_size
    # Reserve additional shared memory for compiler-generated state.
    return staged_inputs + output_tile + 512


def _get_max_shared_memory_bytes() -> int:
    """Return the per-block shared-memory limit."""
    override = os.environ.get("TLE_MAX_SHARED_MEMORY_BYTES", "").strip()
    if override:
        try:
            value = int(override)
        except ValueError as exc:
            raise ValueError(f"TLE_MAX_SHARED_MEMORY_BYTES must be an integer, got {override!r}") from exc
        if value <= 0:
            raise ValueError("TLE_MAX_SHARED_MEMORY_BYTES must be positive")
        return value

    device = torch.cuda.current_device()
    # Triton queries CUDA's per-block opt-in limit.
    try:
        properties = triton.runtime.driver.active.utils.get_device_properties(device)
        value = properties.get("max_shared_mem")
        if value is not None and int(value) > 0:
            return int(value)
    except (AttributeError, RuntimeError):
        pass

    properties = torch.cuda.get_device_properties(device)
    for attribute in ("shared_memory_per_block_optin", "shared_memory_per_block"):
        value = getattr(properties, attribute, None)
        if value is not None and int(value) > 0:
            return int(value)

    raise RuntimeError("cannot determine per-block shared-memory limit; set "
                       "TLE_MAX_SHARED_MEMORY_BYTES explicitly")


def _get_joint_gemm_candidates(
    M: int,
    N: int,
    K: int,
    world_size: int,
    base_config: TleGemmConfig,
) -> list[TleGemmConfig]:
    """Filter GEMM configurations by alignment and shared-memory usage."""
    epilogue_setting = _get_epilogue_subtile_setting()
    global_epilogue_choices = ((False, True) if epilogue_setting is None else (epilogue_setting, ))
    configs = [dataclasses.replace(base_config, epilogue_subtile=epilogue) for epilogue in global_epilogue_choices]

    if _env_bool("TLE_GEMM_AUTOTUNE", True):
        raw = os.environ.get("TLE_GEMM_AUTOTUNE_CONFIGS", "").strip()
        tuples = []
        if raw:
            # BM:BN:BK:GROUP_M:WARPS:STAGES[:EPILOGUE],...
            for item in raw.split(","):
                fields = item.strip().split(":")
                if len(fields) not in (6, 7):
                    raise ValueError("TLE_GEMM_AUTOTUNE_CONFIGS entries must be "
                                     "BM:BN:BK:GROUP_M:WARPS:STAGES[:EPILOGUE]")
                values = [int(value) for value in fields[:6]]
                candidate_epilogue = None
                if len(fields) == 7:
                    normalized = fields[6].strip().lower()
                    if normalized in ("1", "true", "yes", "on"):
                        candidate_epilogue = True
                    elif normalized in ("0", "false", "no", "off"):
                        candidate_epilogue = False
                    else:
                        raise ValueError(f"invalid candidate EPILOGUE value {fields[6]!r}")
                tuples.append((*values, candidate_epilogue))
        else:
            # Built-in candidates include both epilogue variants.
            tuples = _GEMM_AUTOTUNE_CONFIGS

        for block_m, block_n, block_k, group_m, warps, stages, override in tuples:
            if epilogue_setting is not None:
                epilogue_choices = (epilogue_setting, )
            elif override is not None:
                epilogue_choices = (override, )
            else:
                epilogue_choices = (False, True)

            for epilogue in epilogue_choices:
                config = TleGemmConfig(
                    block_m=block_m,
                    block_n=block_n,
                    block_k=block_k,
                    group_m=group_m,
                    num_warps=warps,
                    num_stages=stages,
                    epilogue_subtile=epilogue,
                )
                config.validate()
                configs.append(config)

    M_per_rank = M // world_size
    local_K = K // world_size
    max_shared_memory = _get_max_shared_memory_bytes()
    valid = []
    seen = set()
    for config in configs:
        key = _gemm_config_key(config)
        if key in seen:
            continue
        seen.add(key)
        shape_is_legal = (M_per_rank % config.block_m == 0 and N % config.block_n == 0
                          and local_K % config.block_k == 0)
        estimated_shared = _estimated_gemm_shared_memory_bytes(config)
        shared_memory_is_legal = estimated_shared <= max_shared_memory
        if shape_is_legal and shared_memory_is_legal:
            valid.append(config)
        elif shape_is_legal and not shared_memory_is_legal:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(
                    "[Rank 0] skipping GEMM config because estimated shared "
                    f"memory {estimated_shared} exceeds hardware limit "
                    f"{max_shared_memory}: {_gemm_config_dict(config)}",
                    flush=True,
                )
    if not valid:
        raise ValueError("no legal GEMM autotune config fits the shape and shared-memory limit")
    return valid


def _get_min_gemm_sms(total_sms: int) -> tuple[int, float]:
    """Compute the minimum GEMM SM budget."""
    raw_fraction = os.environ.get("TLE_MIN_GEMM_SM_FRACTION", "0.80")
    try:
        fraction = float(raw_fraction)
    except ValueError as exc:
        raise ValueError(f"TLE_MIN_GEMM_SM_FRACTION must be a float, got {raw_fraction!r}") from exc
    if not 0.0 < fraction <= 1.0:
        raise ValueError("TLE_MIN_GEMM_SM_FRACTION must be in (0, 1]")

    fraction_min = math.ceil(total_sms * fraction)
    explicit_min = int(os.environ.get("TLE_MIN_GEMM_SMS", "1"))
    min_gemm_sms = max(fraction_min, explicit_min)
    if min_gemm_sms >= total_sms:
        raise ValueError(f"minimum GEMM SMS must be below total_sms={total_sms}, "
                         f"got {min_gemm_sms}")
    return min_gemm_sms, fraction


def _get_joint_sms_candidates(
    total_sms: int,
    fixed_comm_sms: int,
    min_gemm_sms: int,
) -> list[tuple[int, int]]:
    """Generate communication budgets preserving the GEMM reservation."""
    raw = os.environ.get("TLE_AUTOTUNE_CANDIDATES", "").strip()
    if raw:
        candidates = []
        for item in raw.split(","):
            fields = item.strip().split(":")
            if len(fields) != 2:
                raise ValueError(f"invalid SMS candidate {item!r}; expected scatter:reduction")
            pair = tuple(map(int, fields))
            if min(pair) < 1:
                raise ValueError("scatter and reduction SMS counts must be positive")
            candidates.append(pair)
    else:
        # Search the remaining communication budget densely.
        axis = [1, 2, 4, 6, 8, 10, 12, 16]
        candidates = [(scatter_sms, reduction_sms) for scatter_sms in axis for reduction_sms in axis]

    valid = sorted(
        {pair
         for pair in candidates
         if pair[0] + pair[1] + fixed_comm_sms + min_gemm_sms <= total_sms},
        key=lambda pair: (pair[0] + pair[1], pair[0], pair[1]),
    )
    if not valid:
        raise ValueError(f"no SMS candidate leaves min_gemm_sms={min_gemm_sms}; "
                         f"total_sms={total_sms}, fixed_comm_sms={fixed_comm_sms}")
    return valid


def _representative_sms_candidates(
    candidates: list[tuple[int, int]],
    count: int = 3,
) -> list[tuple[int, int]]:
    """Select representative SM budgets for coarse tuning."""
    if len(candidates) <= count:
        return candidates

    # Sample 40%, 80%, and 100% of the maximum communication budget,
    # favoring a 2:1 scatter/reduction split. Duplicate pairs are removed.
    max_budget = max(scatter + reduction for scatter, reduction in candidates)
    fractions = (0.4, 0.8, 1.0)[:count]
    selected = []
    for fraction in fractions:
        target = round(max_budget * fraction)
        candidate = min(
            candidates,
            key=lambda pair: (
                2 * abs(sum(pair) - target) + abs(pair[0] - 2 * pair[1]),
                abs(sum(pair) - target),
                abs(pair[0] - 2 * pair[1]),
                pair[0] + pair[1],
            ),
        )
        if candidate not in selected:
            selected.append(candidate)
    return selected


def _joint_candidate_key(candidate: dict) -> tuple:
    return (
        _gemm_config_key(candidate["gemm_config"]),
        candidate["scatter_sms"],
        candidate["reduction_sms"],
        tuple(sorted(candidate["rs_config"].items())),
    )


def _benchmark_joint_candidate(
    *,
    ctx: TleGemmReduceScatterContext,
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    TP_GROUP,
    rank: int,
    gemm_config: TleGemmConfig,
    scatter_sms: int,
    reduction_sms: int,
    rs_config: dict,
    warmup: int,
    iters: int,
    clear_l2: bool,
    label: str,
) -> dict:
    """Measure end-to-end latency using per-sample rank maxima."""
    ctx.gemm_config = gemm_config
    _configure_sms(ctx, scatter_sms, reduction_sms)
    torch.cuda.synchronize()
    dist.barrier(group=TP_GROUP)

    score_ms = _time_ms(
        lambda: gemm_rs_multi_node(input_tensor, weight, ctx, rs_config=rs_config),
        torch.cuda.current_stream(),
        warmup=warmup,
        iters=iters,
        clear_l2=clear_l2,
        group=TP_GROUP,
    )
    result = {
        "gemm_config": gemm_config,
        "scatter_sms": scatter_sms,
        "reduction_sms": reduction_sms,
        "rs_config": dict(rs_config),
        "gemm_sms": ctx.num_gemm_sms,
        "p2p_sms": ctx.rs_ctx.num_p2p_sms,
        "sync_sms": ctx.rs_ctx.num_sync_sms,
        "score_ms": score_ms,
    }
    if rank == 0:
        print(
            f"[Rank 0] {label}: GEMM={_gemm_config_dict(gemm_config)}, "
            f"SMS=(gemm={ctx.num_gemm_sms}, scatter={scatter_sms}, "
            f"reduction={reduction_sms}), RS={rs_config}, "
            f"median(max-rank)={score_ms:.3f} ms",
            flush=True,
        )
    return result


def _save_joint_autotune_result(
    *,
    M: int,
    N: int,
    K: int,
    world_size: int,
    local_world_size: int,
    dtype: torch.dtype,
    total_sms: int,
    min_gemm_sms: int,
    min_gemm_fraction: float,
    best_candidate: dict,
    tle_median_ms: float,
    torch_median_ms: float,
    rank: int,
) -> None:
    """Save the winning joint configuration and measurements on rank zero."""
    if rank != 0:
        return

    output_value = os.environ.get(
        "TLE_GEMM_RS_AUTOTUNE_OUTPUT",
        "tle_gemm_rs_autotune_results.json",
    ).strip()
    if not output_value:
        return

    output_path = Path(output_value)
    if output_path.exists():
        data = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        data = {"version": 1, "operator": "tle_gemm_reduce_scatter", "results": {}}
    if not isinstance(data, dict) or not isinstance(data.get("results"), dict):
        raise ValueError(f"invalid autotune result file: {output_path}")

    dtype_name = str(dtype).removeprefix("torch.")
    key = (f"shape={M}x{N}x{K}/world={world_size}/"
           f"local_world={local_world_size}/dtype={dtype_name}")
    best_config = {
        "gemm": _gemm_config_dict(best_candidate["gemm_config"]),
        "sms": {
            "total_sms": total_sms,
            "min_gemm_sms": min_gemm_sms,
            "min_gemm_fraction": min_gemm_fraction,
            "num_gemm_sms": best_candidate["gemm_sms"],
            "num_scatter_sms": best_candidate["scatter_sms"],
            "num_reduction_sms": best_candidate["reduction_sms"],
            "num_p2p_sms": best_candidate["p2p_sms"],
            "num_sync_sms": best_candidate["sync_sms"],
        },
        "reduce_scatter": best_candidate["rs_config"],
    }
    data["results"][key] = {
        "shape": [M, N, K],
        "world_size": world_size,
        "local_world_size": local_world_size,
        "dtype": dtype_name,
        "best_config": best_config,
        "tle_median_ms": tle_median_ms,
        "torch_median_ms": torch_median_ms,
        "speedup": torch_median_ms / tle_median_ms,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)
    print(f"[Rank 0] Saved joint autotune result to {output_path}", flush=True)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Persistent GEMM with hierarchical reduce-scatter.")
    parser.add_argument("--M", "--m", dest="M", type=int, default=2048, help="Global output rows (default: 2048).")
    parser.add_argument("--N", "--n", dest="N", type=int, default=4096, help="Output columns (default: 4096).")
    parser.add_argument("--K", "--k", dest="K", type=int, default=16384,
                        help="Global reduction dimension; each rank uses K/world_size (default: 16384).")
    parser.add_argument("--dtype", type=str.lower, choices=("bf16", "bfloat16", "fp16", "float16"),
                        help="Input dtype; overrides DTYPE (default: bf16).")
    args = parser.parse_args(argv)
    if min(args.M, args.N, args.K) <= 0:
        parser.error("M, N and K must be positive")
    return args


def main():
    args = _parse_args()
    M, N, K = args.M, args.N, args.K
    dtype = (get_test_dtype() if args.dtype is None else {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }[args.dtype])

    tle.get_mem_pool()
    rank = dist.get_rank()
    TP_GROUP = dist.group.WORLD
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    torch.cuda.set_device(local_rank)

    if world_size < 2:
        print("This example needs at least two GPUs", file=sys.stderr)
        return
    if torch.cuda.get_device_capability()[0] < 9:
        print("Skip: persistent TMA GEMM requires sm90 or newer")
        tle.cleanup_communicator()
        return

    validate_node_transfer_shape(M, N, world_size, local_world_size)
    base_gemm_config = _get_gemm_config()
    _validate_problem_size(
        M,
        N,
        K,
        world_size,
        block_m=base_gemm_config.block_m,
        block_n=base_gemm_config.block_n,
        block_k=base_gemm_config.block_k,
    )

    total_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    min_gemm_sms, min_gemm_fraction = _get_min_gemm_sms(total_sms)
    fixed_comm_sms = 1  # Multi-node context currently uses p2p=1, sync=0.
    sms_candidates = _get_joint_sms_candidates(total_sms, fixed_comm_sms, min_gemm_sms)
    initial_scatter_sms, initial_reduction_sms = sms_candidates[0]

    gemm_candidates = _get_joint_gemm_candidates(M, N, K, world_size, base_gemm_config)
    initial_gemm_config = gemm_candidates[0]
    M_per_rank = M // world_size
    valid_rs_configs = [
        config for config in RS_KERNEL_CONFIGS if config["BLOCK_M"] <= M_per_rank and config["BLOCK_N"] <= N
    ]
    if not valid_rs_configs:
        raise ValueError(f"no RS config fits M_per_rank={M_per_rank}, N={N}")

    local_K = K // world_size
    scale = rank + 1
    input_tensor = (torch.rand((M, local_K), dtype=dtype, device="cuda") * (0.02 * scale) - 0.01 * scale)
    weight = (torch.rand((N, local_K), dtype=dtype, device="cuda") * (0.02 * scale) - 0.01 * scale)
    rs_stream = torch.cuda.Stream(priority=-1)
    ctx = create_gemm_rs_context(
        M,
        N,
        rank,
        world_size,
        local_world_size,
        dtype,
        rs_stream,
        gemm_config=initial_gemm_config,
        num_reduction_sms=initial_reduction_sms,
        num_scatter_sms=initial_scatter_sms,
    )

    try:
        fixed_comm_sms = ctx.rs_ctx.num_p2p_sms + ctx.rs_ctx.num_sync_sms
        sms_candidates = _get_joint_sms_candidates(total_sms, fixed_comm_sms, min_gemm_sms)
        representative_sms = _representative_sms_candidates(sms_candidates)

        # Use short tuning runs, then longer finalist and baseline measurements.
        warmup = _env_positive_int("TLE_AUTOTUNE_WARMUP", 3)
        iters = _env_positive_int("TLE_AUTOTUNE_ITERS", 10)
        gemm_top_k = _env_positive_int("TLE_GEMM_TOP_K", 4)
        sms_top_k = _env_positive_int("TLE_SMS_TOP_K", 2)
        final_top_k = _env_positive_int("TLE_FINAL_TOP_K", 5)
        final_warmup = _env_positive_int("TLE_FINAL_WARMUP", 20)
        final_iters = _env_positive_int("TLE_FINAL_ITERS", 200)
        clear_l2 = _env_bool("TLE_AUTOTUNE_CLEAR_L2", False)
        final_clear_l2 = _env_bool("TLE_FINAL_CLEAR_L2", clear_l2)

        if rank == 0:
            print(
                f"[Rank 0] joint autotune "
                f"shape=({M}, {N}, {K}), dtype={dtype}, local_K={local_K}, "
                f"total_sms={total_sms}, min_gemm_sms={min_gemm_sms} "
                f"({min_gemm_fraction:.1%}), fixed_comm_sms={fixed_comm_sms}",
                flush=True,
            )
            print(
                f"[Rank 0] candidates: GEMM={len(gemm_candidates)}, "
                f"SMS={len(sms_candidates)} {sms_candidates}, "
                f"representative_SMS={representative_sms}, "
                f"RS={len(valid_rs_configs)}",
                flush=True,
            )

        def check_correctness(rs_config, stage="correctness"):
            tle_output = gemm_rs_multi_node(input_tensor, weight, ctx, rs_config=rs_config)
            torch.cuda.synchronize()
            _assert_close_on_all_ranks(
                tle_output,
                torch_output,
                shape=(M, N, K),
                stage=stage,
                atol=6e-2,
                rtol=6e-2,
            )

        ctx.gemm_config = initial_gemm_config
        _configure_sms(ctx, *sms_candidates[0])
        torch_output = torch_gemm_rs(input_tensor, weight, TP_GROUP)
        check_correctness(valid_rs_configs[0], stage="initial")
        if rank == 0:
            print("[Rank 0] initial end-to-end correctness PASSED", flush=True)

        def benchmark_candidates(candidates, label, *, final=False):
            candidates = list(candidates)
            results = []
            for index, (gemm_config, scatter_sms, reduction_sms, rs_config) in enumerate(candidates, 1):
                results.append(
                    _benchmark_joint_candidate(
                        ctx=ctx,
                        input_tensor=input_tensor,
                        weight=weight,
                        TP_GROUP=TP_GROUP,
                        rank=rank,
                        gemm_config=gemm_config,
                        scatter_sms=scatter_sms,
                        reduction_sms=reduction_sms,
                        rs_config=rs_config,
                        warmup=final_warmup if final else warmup,
                        iters=final_iters if final else iters,
                        clear_l2=final_clear_l2 if final else clear_l2,
                        label=f"{label} [{index}/{len(candidates)}]",
                    ))
            return results

        def top_k(results, count):
            return sorted(results, key=lambda result: result["score_ms"])[:count]

        # Stage 1: rank GEMM configurations using representative SM budgets and fixed RS.
        stage1 = []
        for index, gemm_config in enumerate(gemm_candidates, 1):
            results = benchmark_candidates(
                ((gemm_config, scatter, reduction, valid_rs_configs[0]) for scatter, reduction in representative_sms),
                f"GEMM coarse {index}/{len(gemm_candidates)}",
            )
            stage1.extend(top_k(results, 1))
        selected_gemm = top_k(stage1, gemm_top_k)
        if rank == 0:
            print(f"[Rank 0] Stage 1 selected {len(selected_gemm)} GEMM configs", flush=True)

        # Stage 2: tune SM budgets for the retained GEMM configurations.
        stage2 = []
        for index, retained in enumerate(selected_gemm, 1):
            results = benchmark_candidates(
                ((retained["gemm_config"], scatter, reduction, valid_rs_configs[0])
                 for scatter, reduction in sms_candidates),
                f"SMS search {index}/{len(selected_gemm)}",
            )
            stage2.extend(top_k(results, sms_top_k))

        # Stage 3: tune RS configurations and retain unique finalists.
        stage3 = benchmark_candidates(
            ((retained["gemm_config"], retained["scatter_sms"], retained["reduction_sms"], rs_config)
             for retained in stage2
             for rs_config in valid_rs_configs),
            "RS search",
        )
        unique = {}
        for result in stage3:
            key = _joint_candidate_key(result)
            if key not in unique or result["score_ms"] < unique[key]["score_ms"]:
                unique[key] = result
        finalists = top_k(unique.values(), final_top_k)

        # Stage 4: remeasure finalists and select the lowest end-to-end latency.
        final_results = benchmark_candidates(
            ((candidate["gemm_config"], candidate["scatter_sms"], candidate["reduction_sms"], candidate["rs_config"])
             for candidate in finalists),
            "Final re-rank",
            final=True,
        )
        best = top_k(final_results, 1)[0]
        ctx.gemm_config = best["gemm_config"]
        _configure_sms(ctx, best["scatter_sms"], best["reduction_sms"])

        # Validate the winner; intermediate candidates are not checked individually.
        check_correctness(best["rs_config"], stage="post_rerank")

        tle_ms = best["score_ms"]
        torch_ms = _time_ms(
            lambda: torch_gemm_rs(input_tensor, weight, TP_GROUP),
            torch.cuda.current_stream(),
            warmup=final_warmup,
            iters=final_iters,
            clear_l2=final_clear_l2,
            group=TP_GROUP,
        )

        check_correctness(best["rs_config"], stage="post_torch_benchmark")

        if rank == 0:
            best_config_for_print = {
                "gemm": _gemm_config_dict(best["gemm_config"]),
                "sms": {
                    "gemm": best["gemm_sms"],
                    "scatter": best["scatter_sms"],
                    "reduction": best["reduction_sms"],
                    "p2p": ctx.rs_ctx.num_p2p_sms,
                    "sync": ctx.rs_ctx.num_sync_sms,
                },
                "reduce_scatter": best["rs_config"],
            }
            print(
                f"[Rank 0] correctness=PASSED best_config="
                f"{best_config_for_print} TLE median={tle_ms:.3f}ms "
                f"Torch median={torch_ms:.3f}ms "
                f"speedup={torch_ms / tle_ms:.2f}x",
                flush=True,
            )

        _save_joint_autotune_result(
            M=M,
            N=N,
            K=K,
            world_size=world_size,
            local_world_size=local_world_size,
            dtype=dtype,
            total_sms=total_sms,
            min_gemm_sms=min_gemm_sms,
            min_gemm_fraction=min_gemm_fraction,
            best_candidate=best,
            tle_median_ms=tle_ms,
            torch_median_ms=torch_ms,
            rank=rank,
        )
        dist.barrier(group=TP_GROUP)
    finally:
        ctx.finalize()


if __name__ == "__main__":
    main()
