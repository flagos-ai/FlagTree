"""Hierarchical inter-node reduce-scatter.

Run once on each node from the repository root:

    # Node 0
    NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 NPROC_PER_NODE=4 \
        bash python/tutorials/tle/07-inter-node-reduce-scatter.sh
    # Node 1
    NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 NPROC_PER_NODE=4 \
        bash python/tutorials/tle/07-inter-node-reduce-scatter.sh

Use matching topology, dimensions, dtype, and tuning settings on every node.
MASTER_ADDR must be reachable from all nodes; MASTER_PORT defaults to 29501.

SHAPES accepts one MxN pair (default: 2048x4096); DTYPE defaults to bf16.
NPROC_PER_NODE defaults to gpu (one process per visible GPU).
Set CUDA_VISIBLE_DEVICES externally to select GPUs.
FLAGCX_IB_HCA is configured in the launch script.
Requires SM90 or newer and matching FlagTree/FlagCX runtime and device bitcode.
"""

from __future__ import annotations

import dataclasses
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import triton
from triton.runtime import DistributedRtContext
import triton.language as tl
import triton.experimental.tle.language as tle


def validate_node_transfer_shape(M: int, N: int, world_size: int, local_world_size: int) -> None:
    """Validate topology, shard dimensions, and node-transfer size."""
    if M <= 0 or N <= 0:
        raise ValueError("M and N must be positive")
    if world_size < 2 or local_world_size < 1 or world_size % local_world_size:
        raise ValueError("world_size must be >= 2 and divisible by positive local_world_size")
    if M % world_size:
        raise ValueError(f"M={M} must be divisible by world_size={world_size}")
    if world_size == local_world_size:
        return  # No node-space arange on the single-node path.
    shard_elements = (M // world_size) * N
    block_size = triton.next_power_of_2(shard_elements)
    limit = tl.TRITON_MAX_TENSOR_NUMEL
    if block_size > limit:
        raise ValueError(f"node transfer shape M={M}, N={N}, world_size={world_size} "
                         f"has {shard_elements} elements per rank; "
                         f"tl.arange BLOCK_SIZE=next_power_of_2((M/world_size)*N)={block_size} "
                         f"exceeds maxTensorNumElements={limit}. "
                         "Reduce M or N via SHAPES (RS) or --M/--N (GEMM+RS); "
                         "changing K does not reduce the transfer size.")


@dataclasses.dataclass
class TleReduceScatter2DContext:

    max_M: int
    N: int
    rank: int
    world_size: int
    local_world_size: int
    dtype: torch.dtype
    with_gemm_output: bool
    # Scatter, reduction, and P2P share one registered communication buffer.
    comm_buf: torch.Tensor
    gemm_out_buf: Optional[torch.Tensor]
    scatter_buf: torch.Tensor
    rs_per_node_buf: torch.Tensor
    p2p_buf: torch.Tensor
    signal_buf: torch.Tensor
    dist_ctx: DistributedRtContext

    reduction_stream: torch.cuda.Stream
    num_sync_sms: int
    num_p2p_sms: int
    num_reduction_sms: int
    num_scatter_sms: int

    scatter_signal_buf: torch.Tensor = dataclasses.field(init=False)
    local_rank: int = dataclasses.field(init=False)
    node_id: int = dataclasses.field(init=False)
    nnodes: int = dataclasses.field(init=False)
    device_mesh: tle.device_mesh = dataclasses.field(init=False)
    world_mesh: tle.device_mesh = dataclasses.field(init=False)
    _finalized: bool = dataclasses.field(init=False, default=False)

    def __post_init__(self):
        if self.world_size < 2:
            raise ValueError("TLE reduce-scatter requires at least two GPUs")
        if self.local_world_size < 1 or self.world_size % self.local_world_size:
            raise ValueError("world_size must be divisible by positive local_world_size")
        if self.max_M % self.world_size:
            raise ValueError("max_M must be divisible by world_size")
        self.local_rank = self.rank % self.local_world_size
        self.node_id = self.rank // self.local_world_size
        self.nnodes = self.world_size // self.local_world_size
        self.device_mesh = tle.device_mesh(tle.MeshConfig(device=self.local_world_size))
        self.world_mesh = tle.device_mesh(tle.MeshConfig(node=self.nnodes, device=self.local_world_size))
        if self.signal_buf.numel() < self.world_size:
            raise ValueError("signal_buf must contain one GEMM-ready flag per rank")
        if self.num_scatter_sms < 1:
            raise ValueError("num_scatter_sms must be positive")

        # GEMM-ready flags; tile counters use a separate workspace.
        self.scatter_signal_buf = self.signal_buf[:self.world_size]

    @property
    def num_rs_sms(self) -> int:
        """Return the communication SM budget."""
        if self.nnodes == 1:
            return self.num_scatter_sms
        return (self.num_scatter_sms + self.num_sync_sms + self.num_p2p_sms + self.num_reduction_sms)

    def finalize(self):

        if self._finalized:
            return
        torch.cuda.synchronize()
        tle.cleanup_communicator()
        self._finalized = True

    @property
    def rs_per_node_offset_elems(self) -> int:
        return self.max_M * self.N

    @property
    def p2p_offset_elems(self) -> int:
        per_node_elems = (self.max_M // self.local_world_size) * self.N
        return self.rs_per_node_offset_elems + per_node_elems

    def scatter_view(self, M: int, N: int) -> torch.Tensor:
        nelems = M * N
        if nelems > self.rs_per_node_offset_elems:
            raise ValueError("shape exceeds the registered scatter buffer")
        return self.comm_buf[:nelems].view(M, N)

    def rs_per_node_view(self, M: int, N: int) -> torch.Tensor:
        rows = M // self.local_world_size
        nelems = rows * N
        start = self.rs_per_node_offset_elems
        if start + nelems > self.p2p_offset_elems:
            raise ValueError("shape exceeds the registered per-node buffer")
        return self.comm_buf[start:start + nelems].view(rows, N)

    def p2p_view(self, M: int, N: int) -> torch.Tensor:
        rows = M // self.local_world_size
        nelems = rows * N
        start = self.p2p_offset_elems
        if start + nelems > self.comm_buf.numel():
            raise ValueError("shape exceeds the registered p2p buffer")
        return self.comm_buf[start:start + nelems].view(rows, N)

    def reset_barriers(self):
        self.signal_buf.zero_()


@triton.jit
def _scatter_kernel(
    input_ptr,
    local_scatter_ptr,
    dist_ctx: tl.constexpr,
    ready_ptr,
    M_per_rank,
    N,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    SCATTER_NODE_SLICE_OFFSET_ELEMS: tl.constexpr,
    WAIT_FOR_READY: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Scatter ready shards to local GPUs."""

    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_tiles_m = tl.cdiv(M_per_rank, BLOCK_M)
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    tiles_per_peer = num_tiles_m * num_tiles_n
    row_offs = tl.arange(0, BLOCK_M)
    col_offs = tl.arange(0, BLOCK_N)

    source_slot_offset_elems = LOCAL_RANK * M_per_rank * N
    for step in range(WORLD_SIZE):
        target_rank = (LOCAL_RANK + step + 1) % WORLD_SIZE

        # Wait for this GPU to finish the destination GEMM shard.
        if WAIT_FOR_READY:
            while tl.atomic_add(ready_ptr + target_rank, 0, sem="acquire", scope="gpu") == 0:
                pass

        if target_rank == LOCAL_RANK:
            remote_base = local_scatter_ptr + source_slot_offset_elems
        else:
            remote_base = tle.remote(
                dist_ctx,
                space="device",
                dtype=input_ptr.dtype.element_ty,
                shard_id=target_rank,
                offset=SCATTER_NODE_SLICE_OFFSET_ELEMS + source_slot_offset_elems,
            )

        for local_tile in range(pid, tiles_per_peer, num_pid):
            tile_m = local_tile // num_tiles_n
            tile_n = local_tile % num_tiles_n
            input_row = target_rank * M_per_rank + tile_m * BLOCK_M
            input_col = tile_n * BLOCK_N
            input_ptrs = (input_ptr + (input_row + row_offs[:, None]) * N + input_col + col_offs[None, :])
            row_mask = (input_row + row_offs[:, None]) < (target_rank + 1) * M_per_rank
            col_mask = (input_col + col_offs[None, :]) < N
            values = tl.load(input_ptrs, mask=row_mask & col_mask, other=0.0)

            scatter_row = tile_m * BLOCK_M
            scatter_ptrs = (remote_base + (scatter_row + row_offs[:, None]) * N + input_col + col_offs[None, :])
            scatter_row_mask = (scatter_row + row_offs[:, None]) < M_per_rank
            tl.store(scatter_ptrs, values, mask=scatter_row_mask & col_mask)


@triton.jit
def _device_barrier_kernel(dist_ctx: tl.constexpr, mesh: tl.constexpr):
    """Synchronize local GPUs before reduction."""

    tle.distributed_barrier(
        mesh=mesh,
        device_dptr=dist_ctx,
        space="device",
        group_kind="block",
        barrier_kind="sync",
        order="acqrel",
        index=0,
    )


@triton.jit
def _world_barrier_kernel(dist_ctx: tl.constexpr, mesh: tl.constexpr):
    tle.distributed_barrier(
        mesh=mesh,
        device_dptr=dist_ctx,
        space="world",
        group_kind="block",
        barrier_kind="sync",
        order="acqrel",
        index=0,
    )


@triton.jit
def _inter_node_p2p_kernel(
    comm_buf,
    offset,
    local_world_size,
    M_per_rank,
    N,
    ctx: tl.constexpr,
    DTYPE: tl.constexpr,
    NODE_ID: tl.constexpr,
    NNODES: tl.constexpr,
    LOCAL_RANK: tl.constexpr,
    RS_OFFSET: tl.constexpr,
    P2P_OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Send a locally reduced shard to its destination node."""
    remote_node_id = (offset + 1 + NODE_ID) % NNODES
    remote_rank = LOCAL_RANK + remote_node_id * local_world_size
    nelems_per_rank = M_per_rank * N

    # Reduction slots are indexed by destination node; receive slots by source node.
    src_offset = RS_OFFSET + remote_node_id * nelems_per_rank
    dst_offset = P2P_OFFSET + NODE_ID * nelems_per_rank

    remote_dst = tle.remote(
        ctx,
        space="node",
        dtype=DTYPE,
        shard_id=remote_rank,
        coopkind=tle.GroupKind.BLOCK,
    )

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < nelems_per_rank
    values = tl.load(
        comm_buf + src_offset + offsets,
        mask=mask,
    )
    tl.store(
        remote_dst + dst_offset + offsets,
        values,
        mask=mask,
    )


def _tl_dtype(dtype: torch.dtype) -> tl.dtype:
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.float32:
        return tl.float32
    raise ValueError(f"unsupported TLE node-put dtype: {dtype}")


# All ranks evaluate the same candidates and select from aggregated timings.
KERNEL_CONFIGS = [
    {"BLOCK_M": 128, "BLOCK_N": 64, "scatter_num_warps": 4, "reduce_num_warps": 4, "final_num_warps": 4},
    {"BLOCK_M": 128, "BLOCK_N": 128, "scatter_num_warps": 4, "reduce_num_warps": 4, "final_num_warps": 4},
    {"BLOCK_M": 256, "BLOCK_N": 64, "scatter_num_warps": 4, "reduce_num_warps": 4, "final_num_warps": 4},
    {"BLOCK_M": 256, "BLOCK_N": 128, "scatter_num_warps": 4, "reduce_num_warps": 4, "final_num_warps": 4},
    {"BLOCK_M": 256, "BLOCK_N": 128, "scatter_num_warps": 8, "reduce_num_warps": 8, "final_num_warps": 8},
]

# Use SHAPES=MxN to override the default two-node, eight-GPU shape.
DEFAULT_SHAPES = [(2048, 4096)]


def _get_shapes():
    """Parse SHAPES=MxN; default to 2048x4096."""
    value = os.environ.get("SHAPES", "").strip()
    if not value:
        return list(DEFAULT_SHAPES)
    if "," in value:
        raise ValueError("Run exactly one SHAPES=MxN entry per torchrun process. "
                         "The node-space communication window cannot safely switch packed "
                         "layouts between shapes; launch a separate process for each shape.")
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError("SHAPES must use MxN, for example 2048x4096")
    try:
        M, N = map(int, parts)
    except ValueError as exc:
        raise ValueError("SHAPES must contain integer M and N, for example 2048x4096") from exc
    if M <= 0 or N <= 0:
        raise ValueError("all SHAPES dimensions must be positive")
    return [(M, N)]


def get_test_dtype() -> torch.dtype:
    """Parse the shared BF16/FP16 benchmark dtype."""
    value = os.environ.get("DTYPE", "bf16").strip().lower()
    options = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    if value not in options:
        raise ValueError(f"unsupported DTYPE={value!r}; use bf16 or fp16")
    return options[value]


def _env_positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _get_sms_candidates(total_sms: int, fixed_comm_sms: int) -> list[tuple[int, int]]:
    """Generate scatter/reduction allocations using all available SMs."""
    raw = os.environ.get("TLE_AUTOTUNE_CANDIDATES", "").strip()
    if raw:
        candidates = []
        seen = set()
        for item in raw.split(","):
            fields = item.strip().split(":")
            if len(fields) != 2:
                raise ValueError(f"invalid SMS candidate {item!r}; expected scatter:reduction")
            pair = (int(fields[0]), int(fields[1]))
            if pair[0] < 1 or pair[1] < 1:
                raise ValueError("SMS candidates must be positive")
            if pair not in seen:
                candidates.append(pair)
                seen.add(pair)
        if not candidates:
            raise ValueError("TLE_AUTOTUNE_CANDIDATES must not be empty")
        return candidates

    allocatable_sms = total_sms - fixed_comm_sms
    if allocatable_sms < 2:
        raise ValueError("at least two SMs must remain for scatter and reduction")

    # Test both scatter/reduction splits within the available SM budget.
    sms_axis = {1, 2, 4, 8, 12, 16}
    sms_axis.update(range(8, allocatable_sms, 8))
    sms_axis.update({
        allocatable_sms // 3,
        allocatable_sms // 2,
        (3 * allocatable_sms) // 4,
        allocatable_sms - 2,
    })
    sms_axis = sorted(value for value in sms_axis if 0 < value < allocatable_sms)
    candidates = set()
    for value in sms_axis:
        complement = allocatable_sms - value
        candidates.add((value, complement))
        candidates.add((complement, value))
    return sorted(candidates)


def _configure_sms(
    ctx: TleReduceScatter2DContext,
    num_scatter_sms: int,
    num_reduction_sms: int,
) -> None:
    """Apply scatter and reduction SM budgets."""
    if num_scatter_sms < 1 or num_reduction_sms < 1:
        raise ValueError("scatter and reduction SMS counts must be positive")
    ctx.num_scatter_sms = num_scatter_sms
    ctx.num_reduction_sms = num_reduction_sms


def _save_autotune_result(
    *,
    M: int,
    N: int,
    world_size: int,
    local_world_size: int,
    dtype: torch.dtype,
    best_config: dict,
    tle_median_ms: float,
    torch_median_ms: float,
    rank: int,
) -> None:
    """Save the selected configuration and measurements on rank zero."""
    if rank != 0:
        return
    output_value = os.environ.get("TLE_AUTOTUNE_OUTPUT", "tle_rs_autotune_results.json").strip()
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
    key = (f"shape={M}x{N}/world={world_size}/"
           f"local_world={local_world_size}/dtype={dtype_name}")
    data["results"][key] = {
        "shape": [M, N],
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
    print(f"Saved autotune result to {output_path}", flush=True)


@triton.jit
def _ring_reduce_tma_kernel(
    local_scatter_ptr,
    output_ptr,
    M_per_rank,
    N,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Sum shard contributions using TMA loads."""

    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_tiles_m = tl.cdiv(M_per_rank, BLOCK_M)
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_tiles_m * num_tiles_n
    scatter_desc = tl.make_tensor_descriptor(
        local_scatter_ptr,
        shape=[M_per_rank * WORLD_SIZE, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    output_desc = tl.make_tensor_descriptor(
        output_ptr,
        shape=[M_per_rank, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    for tile_id in range(pid, total_tiles, num_pid):
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id % num_tiles_n
        row = tile_m * BLOCK_M
        col = tile_n * BLOCK_N
        source_rank = (LOCAL_RANK + 1) % WORLD_SIZE
        accum = scatter_desc.load([row + source_rank * M_per_rank, col])

        for i in range(1, WORLD_SIZE):
            source_rank = (LOCAL_RANK + i + 1) % WORLD_SIZE
            accum += scatter_desc.load([row + source_rank * M_per_rank, col])
        output_desc.store([row, col], accum)


def create_tle_reduce_scatter_2d_ctx(max_M: int, N: int, rank: int, world_size: int, local_world_size: int,
                                     dtype: torch.dtype, with_gemm_output: bool = False,
                                     reduction_stream: Optional[torch.cuda.Stream] = None, num_reduction_sms: int = 15,
                                     num_scatter_sms: int = 16) -> TleReduceScatter2DContext:
    """Allocate and register the hierarchical reduce-scatter buffers."""

    validate_node_transfer_shape(max_M, N, world_size, local_world_size)

    per_node_rows = max_M // local_world_size
    scatter_elems = max_M * N
    per_node_elems = per_node_rows * N
    comm_elems = scatter_elems + 2 * per_node_elems
    with torch.cuda.use_mem_pool(tle.get_mem_pool()):
        gemm_out_buf = (torch.empty((max_M, N), dtype=dtype, device="cuda") if with_gemm_output else None)
        comm_buf = torch.empty((comm_elems, ), dtype=dtype, device="cuda")
        scatter_buf = comm_buf[:scatter_elems].view(max_M, N)
        rs_start = scatter_elems
        rs_per_node_buf = comm_buf[rs_start:rs_start + per_node_elems].view(per_node_rows, N)
        p2p_start = rs_start + per_node_elems
        p2p_buf = comm_buf[p2p_start:p2p_start + per_node_elems].view(per_node_rows, N)
        signal_buf = torch.empty((world_size, ), dtype=torch.int32, device="cuda")
    signal_buf.zero_()

    dist_ctx = tle.create_dist_tensor(comm_buf)
    return TleReduceScatter2DContext(
        max_M=max_M,
        N=N,
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        dtype=dtype,
        with_gemm_output=with_gemm_output,
        gemm_out_buf=gemm_out_buf,
        comm_buf=comm_buf,
        scatter_buf=scatter_buf,
        rs_per_node_buf=rs_per_node_buf,
        p2p_buf=p2p_buf,
        signal_buf=signal_buf,
        dist_ctx=dist_ctx,
        reduction_stream=(reduction_stream if reduction_stream is not None else torch.cuda.Stream(priority=-1)),
        num_sync_sms=0,
        num_p2p_sms=1,
        num_reduction_sms=num_reduction_sms,
        num_scatter_sms=num_scatter_sms,
    )


def _set_tma_allocator():
    """Set the allocator for TMA descriptor storage."""

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)


def reduce_scatter_for_each_node(input_tensor: torch.Tensor, stream: torch.cuda.Stream, ctx: TleReduceScatter2DContext,
                                 ready_flags: Optional[torch.Tensor] = None,
                                 config: Optional[dict] = None) -> torch.Tensor:
    """Reduce locally for each destination node and launch transfers."""

    if config is None:
        config = KERNEL_CONFIGS[0]

    world_size = ctx.world_size
    local_world_size = ctx.local_world_size
    local_rank = ctx.local_rank
    reduction_stream = ctx.reduction_stream
    num_reduction_sms = ctx.num_reduction_sms
    nnodes = ctx.nnodes
    node_id = ctx.node_id
    M, N = input_tensor.shape
    scatter_buf = ctx.scatter_view(M, N)
    rs_per_node_buf = ctx.rs_per_node_view(M, N)
    p2p_buf = ctx.p2p_view(M, N)
    M_per_rank = M // world_size
    M_per_node = M_per_rank * local_world_size

    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    scatter_num_warps = config["scatter_num_warps"]
    reduce_num_warps = config["reduce_num_warps"]

    scatter_grid = lambda META: (min(
        triton.cdiv(M_per_rank, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        ctx.num_scatter_sms,
    ), )

    def reduce_launch_config(num_sms: int):
        if num_sms == -1:
            return (lambda META: (triton.cdiv(M_per_rank, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )
                    ), BLOCK_N, reduce_num_warps
        return (lambda META:
                (min(triton.cdiv(M_per_rank, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), num_sms), )
                ), BLOCK_N, reduce_num_warps

    # Only fused GEMM waits for per-rank ready flags.

    with torch.cuda.stream(stream):
        for n in range(nnodes):
            # Process remote destinations first and this node last.
            cur_node_id = (node_id + n + 1) % nnodes

            input_intra_node = input_tensor[cur_node_id * M_per_node:(cur_node_id + 1) * M_per_node]

            scatter_for_node = scatter_buf[cur_node_id * M_per_node:(cur_node_id + 1) * M_per_node]

            rs_per_node_output = rs_per_node_buf[cur_node_id * M_per_rank:(cur_node_id + 1) * M_per_rank]

            # Ready flags use global-rank order.

            signal_start = cur_node_id * local_world_size

            if ready_flags is not None:
                ready_for_node = ready_flags[signal_start:signal_start + local_world_size]
            else:
                ready_for_node = input_tensor

            # Select the destination node within the registered window; offsets are in elements.
            scatter_node_slice_offset_elems = cur_node_id * M_per_node * N

            _scatter_kernel[scatter_grid](
                input_intra_node,
                scatter_for_node,
                ready_ptr=ready_for_node,
                M_per_rank=M_per_rank,
                N=N,
                dist_ctx=ctx.dist_ctx,
                LOCAL_RANK=local_rank,
                WORLD_SIZE=local_world_size,
                SCATTER_NODE_SLICE_OFFSET_ELEMS=scatter_node_slice_offset_elems,
                WAIT_FOR_READY=ready_flags is not None,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                num_warps=scatter_num_warps,
            )

            # Complete local scatter writes before reduction.
            _device_barrier_kernel[(1, )](dist_ctx=ctx.dist_ctx, mesh=ctx.device_mesh)

            # Use the full reduction grid for the final destination node.
            node_reduce_sms = (-1 if n == nnodes - 1 else num_reduction_sms)

            reduce_grid, reduce_block_n, reduce_warps = reduce_launch_config(node_reduce_sms)

            reduction_stream.wait_stream(stream)

            with torch.cuda.stream(reduction_stream):
                _ring_reduce_tma_kernel[reduce_grid](
                    scatter_for_node,
                    rs_per_node_output,
                    M_per_rank,
                    N,
                    LOCAL_RANK=local_rank,
                    WORLD_SIZE=local_world_size,
                    BLOCK_M=256,
                    BLOCK_N=reduce_block_n,
                    num_warps=reduce_warps,
                )

                if nnodes > 1:
                    if n == nnodes - 1:
                        p2p_buf[node_id * M_per_rank:(node_id + 1) * M_per_rank].copy_(rs_per_node_output)
                    else:
                        _inter_node_p2p_kernel[(ctx.num_p2p_sms, )](
                            ctx.comm_buf,
                            n,
                            local_world_size,
                            M_per_rank,
                            N,
                            ctx=ctx.dist_ctx,
                            DTYPE=_tl_dtype(ctx.dtype),
                            NODE_ID=node_id,
                            NNODES=nnodes,
                            LOCAL_RANK=local_rank,
                            RS_OFFSET=ctx.rs_per_node_offset_elems,
                            P2P_OFFSET=ctx.p2p_offset_elems,
                            BLOCK_SIZE=triton.next_power_of_2(M_per_rank * N),
                            num_warps=16,
                        )

    stream.wait_stream(reduction_stream)
    if nnodes == 1:
        return rs_per_node_buf[:M_per_rank * nnodes]
    return p2p_buf[:M_per_rank * nnodes]


def reduce_scatter_multi_node(input_tensor: torch.Tensor, stream: torch.cuda.Stream, ctx: TleReduceScatter2DContext,
                              output: torch.Tensor, ready_flags: Optional[torch.Tensor] = None,
                              config: Optional[dict] = None) -> torch.Tensor:
    """Combine node contributions into the final output shard."""

    if config is None:
        config = KERNEL_CONFIGS[0]

    M, N = input_tensor.shape
    M_per_rank = M // ctx.world_size

    rs_result_per_node = reduce_scatter_for_each_node(input_tensor, stream, ctx, ready_flags, config=config)

    final_grid = lambda META: (triton.cdiv(M_per_rank, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )

    # With one node, the final reduction copies the local result.
    with torch.cuda.stream(stream):
        if ctx.nnodes > 1:
            # Local reduction/P2P precede this barrier; one CTA owns index 0.
            _world_barrier_kernel[(1, )](dist_ctx=ctx.dist_ctx, mesh=ctx.world_mesh)
        _ring_reduce_tma_kernel[final_grid](
            rs_result_per_node,
            output,
            M_per_rank,
            N,
            LOCAL_RANK=ctx.node_id,
            WORLD_SIZE=ctx.nnodes,
            BLOCK_M=config["BLOCK_M"],
            BLOCK_N=config["BLOCK_N"],
            num_warps=config["final_num_warps"],
        )
    return output


def reduce_scatter_2d_op(input_tensor: torch.Tensor, ctx: TleReduceScatter2DContext,
                         output: Optional[torch.Tensor] = None, ready_flags: Optional[torch.Tensor] = None,
                         config: Optional[dict] = None) -> torch.Tensor:
    """Validate inputs and run hierarchical reduce-scatter."""

    if config is None:
        config = KERNEL_CONFIGS[0]

    M, N = input_tensor.shape
    validate_node_transfer_shape(M, N, ctx.world_size, ctx.local_world_size)
    if input_tensor.dtype != ctx.dtype or N > ctx.N:
        raise ValueError("input shape/dtype exceeds reduce-scatter context")
    if M > ctx.max_M:
        raise ValueError("M exceeds reduce-scatter context capacity")
    M_per_rank = M // ctx.world_size
    if M_per_rank < 256:
        raise ValueError("M_per_rank must be >= 256 for the TMA reduce kernel")
    if M_per_rank < config["BLOCK_M"]:
        raise ValueError(f"M_per_rank ({M_per_rank}) must be >= BLOCK_M ({config['BLOCK_M']})")
    if N < config["BLOCK_N"]:
        raise ValueError(f"N ({N}) must be >= BLOCK_N ({config['BLOCK_N']})")
    if output is None:
        output = torch.empty((M_per_rank, N), dtype=input_tensor.dtype, device=input_tensor.device)
    if tuple(output.shape) != (M_per_rank, N):
        raise ValueError("output has an invalid reduce-scatter shape")
    if ready_flags is not None and ready_flags.numel() != ctx.world_size:
        raise ValueError("ready_flags must contain one entry per target rank")

    _set_tma_allocator()
    reduction_stream = ctx.reduction_stream
    scatter_stream = torch.cuda.current_stream()
    if scatter_stream is reduction_stream:
        raise ValueError("scatter_stream and reduction_stream must be distinct")

    with torch.cuda.stream(scatter_stream):
        _device_barrier_kernel[(1, )](dist_ctx=ctx.dist_ctx, mesh=ctx.device_mesh)

    output = reduce_scatter_multi_node(input_tensor, scatter_stream, ctx, output, ready_flags, config=config)

    with torch.cuda.stream(scatter_stream):
        ctx.reset_barriers()
    return output


def torch_rs(input_tensor: torch.Tensor, TP_GROUP) -> torch.Tensor:
    output = torch.empty((input_tensor.shape[0] // TP_GROUP.size(), input_tensor.shape[1]), dtype=input_tensor.dtype,
                         device=input_tensor.device)
    dist.reduce_scatter_tensor(output, input_tensor, group=TP_GROUP)
    return output


def _assert_close_on_all_ranks(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    shape,
    stage: str,
    atol: float = 6e-2,
    rtol: float = 6e-2,
):
    """Raise on all ranks if any result fails validation."""
    abs_diff = (actual.float() - expected.float()).abs()
    tolerance = atol + rtol * expected.float().abs()
    bad_mask = torch.isnan(abs_diff) | (abs_diff > tolerance)
    local_bad_count = bad_mask.sum(dtype=torch.int64)
    local_max_abs_diff = abs_diff.max()

    global_bad_count = local_bad_count.clone()
    global_max_abs_diff = local_max_abs_diff.clone()
    dist.all_reduce(global_bad_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(global_max_abs_diff, op=dist.ReduceOp.MAX)

    if global_bad_count.item() != 0:
        if local_bad_count.item() != 0:
            print(
                f"[Rank {dist.get_rank()}] shape={shape} stage={stage} FAILED "
                f"bad_elements={local_bad_count.item()} "
                f"local_max_abs_diff={local_max_abs_diff.item():.6f}",
                flush=True,
            )
        if dist.get_rank() == 0:
            print(
                f"shape={shape} stage={stage} global FAILED "
                f"bad_elements={global_bad_count.item()} "
                f"global_max_abs_diff={global_max_abs_diff.item():.6f}",
                flush=True,
            )
        # Flush failure diagnostics before other ranks exit.
        dist.barrier()
        raise AssertionError(f"distributed correctness check failed for shape={shape}, stage={stage}")


def benchmark_reduce_scatter_2d(
    input_tensor: torch.Tensor,
    ctx: TleReduceScatter2DContext,
    output: torch.Tensor,
    config: dict,
    warmup: int,
    iters: int,
    benchmark_torch: bool = True,
):
    """Measure TLE and PyTorch reduce-scatter latency."""

    def run_tle():
        reduce_scatter_2d_op(input_tensor, ctx, output=output, config=config)

    def run_torch():
        dist.reduce_scatter_tensor(output, input_tensor, group=dist.group.WORLD)

    def _time_one(fn):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        dist.barrier()
        start.record(torch.cuda.current_stream())
        fn()
        end.record(torch.cuda.current_stream())
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    for _ in range(warmup):
        dist.barrier()
        run_tle()
        if benchmark_torch:
            dist.barrier()
            run_torch()

    tle_times = [_time_one(run_tle) for _ in range(iters)]
    torch_times = ([_time_one(run_torch) for _ in range(iters)] if benchmark_torch else None)

    def _median(times_ms):
        return float(statistics.median(times_ms))

    results = {"tle": {"median_ms": _median(tle_times)}}
    if torch_times is not None:
        results["torch"] = {"median_ms": _median(torch_times)}
    return results


def _select_best_config(configs, benchmark, *, warmup, iters):
    """Select the same lowest-latency configuration on all ranks."""
    if not configs:
        raise ValueError("autotuning requires at least one candidate")
    scores = torch.tensor(
        [benchmark(config, warmup, iters)["tle"]["median_ms"] for config in configs],
        dtype=torch.float32,
        device="cuda",
    )
    # Score each configuration by the maximum rank-local median.
    dist.all_reduce(scores, op=dist.ReduceOp.MAX)
    return configs[int(scores.argmin().item())]


def main():
    (M, N), = _get_shapes()
    dtype = get_test_dtype()
    # Initialize NCCL and the FlagCX runtime.
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
        print("Skip: the TMA reduce kernel requires sm90 or newer")
        tle.cleanup_communicator()
        return

    validate_node_transfer_shape(M, N, world_size, local_world_size)
    if N % 8:
        raise ValueError(f"N={N} must be divisible by 8 for {dtype} TMA row alignment")
    if M // world_size < 256:
        if rank == 0:
            print(f"Skipping shape ({M}, {N}): M_per_rank must be >= 256")
        tle.cleanup_communicator()
        return
    if rank == 0:
        print(f"SHAPES={M}x{N} dtype={dtype}", flush=True)
    total_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    fixed_comm_sms = 1  # num_p2p_sms=1; num_sync_sms=0
    sms_candidates = [
        pair for pair in _get_sms_candidates(total_sms, fixed_comm_sms)
        if pair[0] + pair[1] + fixed_comm_sms == total_sms
    ]
    if not sms_candidates:
        raise ValueError("every standalone SMS candidate must satisfy scatter + reduction "
                         f"+ fixed_comm == total_sms ({total_sms})")
    initial_scatter_sms, initial_reduction_sms = sms_candidates[0]
    if rank == 0:
        print(
            f"SMS autotune candidates ({len(sms_candidates)}): "
            f"{sms_candidates}; total_sms={total_sms}; "
            f"fixed_comm_sms={fixed_comm_sms}; all_sms_allocated=True",
            flush=True,
        )

    ctx = create_tle_reduce_scatter_2d_ctx(M, N, rank, world_size, local_world_size, dtype,
                                           num_reduction_sms=initial_reduction_sms, num_scatter_sms=initial_scatter_sms)

    try:
        M_per_rank = M // world_size
        valid_configs = [c for c in KERNEL_CONFIGS if c["BLOCK_M"] <= M_per_rank and c["BLOCK_N"] <= N]
        if not valid_configs:
            if rank == 0:
                print(f"Skipping shape ({M}, {N}): no config fits")
            return

        sms_reference_config = valid_configs[0]
        max_scatter_ctas = (triton.cdiv(M_per_rank, sms_reference_config["BLOCK_M"]) *
                            triton.cdiv(N, sms_reference_config["BLOCK_N"]))
        # Local reduction uses BLOCK_M=256.
        max_reduction_ctas = (triton.cdiv(M_per_rank, 256) * triton.cdiv(N, sms_reference_config["BLOCK_N"]))
        shape_sms_candidates = [
            pair for pair in sms_candidates if pair[0] <= max_scatter_ctas and pair[1] <= max_reduction_ctas
        ]
        if not shape_sms_candidates:
            raise ValueError(f"no useful SMS candidate fits shape ({M}, {N})")
        if rank == 0:
            print(
                f"shape=({M}, {N}) useful SMS candidates "
                f"({len(shape_sms_candidates)}): {shape_sms_candidates}",
                flush=True,
            )

        if rank == 0:
            print(
                f"shape=({M}, {N}) starting correctness check",
                flush=True,
            )
        input_tensor = torch.rand((M, N), dtype=dtype, device="cuda")
        output = torch.empty((M_per_rank, N), dtype=dtype, device="cuda")
        torch_output = torch_rs(input_tensor, TP_GROUP)
        torch.cuda.synchronize()

        def check_config(config, stage):
            reduce_scatter_2d_op(input_tensor, ctx, output=output, config=config)
            torch.cuda.current_stream().wait_stream(ctx.reduction_stream)
            torch.cuda.synchronize()
            _assert_close_on_all_ranks(output, torch_output, shape=(M, N), stage=stage)

        check_config(valid_configs[0], stage="initial")

        sms_configs = [{
            "num_scatter_sms": scatter_sms,
            "num_reduction_sms": reduction_sms,
        } for scatter_sms, reduction_sms in shape_sms_candidates]
        sms_warmup = _env_positive_int("TLE_AUTOTUNE_WARMUP", 10)
        sms_iters = _env_positive_int("TLE_AUTOTUNE_ITERS", 30)

        def benchmark_config(config, warmup, iters, *, tune_sms=False):
            candidates = sms_configs if tune_sms else valid_configs
            label = "SMS" if tune_sms else "kernel"
            if tune_sms:
                _configure_sms(ctx, config["num_scatter_sms"], config["num_reduction_sms"])
                torch.cuda.synchronize()
                dist.barrier()
            index = candidates.index(config) + 1
            if rank == 0:
                print(f"shape=({M}, {N}) {label} autotune "
                      f"[{index}/{len(candidates)}] starting: {config}", flush=True)
            results = benchmark_reduce_scatter_2d(
                input_tensor,
                ctx,
                output,
                valid_configs[0] if tune_sms else config,
                warmup,
                iters,
                benchmark_torch=False,
            )
            if rank == 0:
                print(
                    f"shape=({M}, {N}) {label} autotune "
                    f"[{index}/{len(candidates)}] completed: "
                    f"TLE median={results['tle']['median_ms']:.3f}ms", flush=True)
            return results

        best_sms_cfg = _select_best_config(
            sms_configs,
            lambda config, warmup, iters: benchmark_config(config, warmup, iters, tune_sms=True),
            warmup=sms_warmup,
            iters=sms_iters,
        )
        _configure_sms(
            ctx,
            best_sms_cfg["num_scatter_sms"],
            best_sms_cfg["num_reduction_sms"],
        )
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print(
                f"shape=({M}, {N}) selected SMS config={best_sms_cfg}",
                flush=True,
            )

        best_kernel_cfg = _select_best_config(
            valid_configs,
            benchmark_config,
            warmup=10,
            iters=30,
        )
        best_cfg = {
            **best_kernel_cfg,
            **best_sms_cfg,
            "num_p2p_sms": ctx.num_p2p_sms,
            "num_sync_sms": ctx.num_sync_sms,
        }

        check_config(best_cfg, stage="selected")

        if rank == 0:
            print(
                f"shape=({M}, {N}) selected best_config={best_cfg}; "
                "final benchmark starting",
                flush=True,
            )
        best_results = benchmark_reduce_scatter_2d(input_tensor, ctx, output, best_cfg, 10, 200)
        global_medians = torch.tensor(
            [best_results["tle"]["median_ms"], best_results["torch"]["median_ms"]],
            dtype=torch.float64,
            device="cuda",
        )
        dist.all_reduce(global_medians, op=dist.ReduceOp.MAX)
        tle_median_ms, torch_median_ms = map(float, global_medians.tolist())

        # Validate communication state after benchmarking.
        check_config(best_cfg, stage="post_benchmark")

        if rank == 0:
            print(f"shape=({M}, {N}) correctness=PASSED "
                  f"best_config={best_cfg} "
                  f"TLE median={tle_median_ms:.3f}ms "
                  f"Torch median={torch_median_ms:.3f}ms "
                  f"speedup={torch_median_ms / tle_median_ms:.2f}x")

        _save_autotune_result(
            M=M,
            N=N,
            world_size=world_size,
            local_world_size=local_world_size,
            dtype=dtype,
            best_config=best_cfg,
            tle_median_ms=tle_median_ms,
            torch_median_ms=torch_median_ms,
            rank=rank,
        )

        torch.cuda.synchronize()
        dist.barrier()
    finally:
        ctx.finalize()


if __name__ == "__main__":
    main()
