"""
Function-level autotuning for intra-node AllGather with TLE remote pointers.

Run in a FlagTree/FlagCX environment on one node:

    bash python/tutorials/tle/05-intra-node-allgather.sh

The launcher selects the GPUs and FlagCX settings. M/N default to 8192/12288;
M must be divisible by the number of ranks. For a small, masked-tail check:

    M=12 N=1003 bash python/tutorials/tle/05-intra-node-allgather.sh

Each rank directly pushes its input to every output buffer, without staging
or relaying through other GPUs. Tune BLOCK, num_warps, and CTAs per peer for
the complete operation, including its two device barriers. All ranks must
call with autotune_pg=dist.group.WORLD, matching the FlagCX communicator;
the tuner aligns candidates and repeat counts and minimizes the slowest
rank's latency. Ordinary per-rank Triton benchmarking is unsafe here.

Two decorated calls validate tuning and cache reuse against torch, followed
by back-to-back executions of the winner with changed inputs. Disk caching
uses FLAGTREE_AUTOTUNE_CACHE_DIR;

If USE_FLAGCX=0, USE_DIST=0, or USE_TLE_DIST=0 disabled distributed support,
reset those settings before running this tutorial.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from flagtree.tune import autotune, to_hashable

CONFIG_SPACE = [{"block": block, "num_warps": num_warps, "ctas_per_peer": ctas_per_peer}
                for block in (1024, 4096, 16384)
                for num_warps in (4, 8)
                for ctas_per_peer in (4, 16, 64)]


@triton.jit
def _all_gather_push_2d_kernel(
    local_ptr,
    ag_ptr,
    dist_ctx: tl.constexpr,
    mesh: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    ELEM_PER_RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    local_rank = tle.shard_id(mesh, "device", device_dptr=dist_ctx)
    # Rotate destinations so ranks do not all start by writing to GPU 0.
    peer = (tl.program_id(0) + local_rank + 1) % WORLD_SIZE
    dst_base = local_rank * ELEM_PER_RANK
    if peer != local_rank:
        dst_ptr = tle.remote(
            dist_ctx,
            shard_id=peer,
            space="device",
            dtype=ag_ptr.dtype.element_ty,
            offset=dst_base,
        )
    else:
        dst_ptr = ag_ptr + dst_base

    for block_id in range(tl.program_id(1), tl.cdiv(ELEM_PER_RANK, BLOCK), tl.num_programs(1)):
        offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < ELEM_PER_RANK
        vals = tl.load(local_ptr + offsets, mask=mask, other=0.0)
        tl.store(dst_ptr + offsets, vals, mask=mask)


@triton.jit
def _all_gather_barrier_kernel(
    dist_ctx: tl.constexpr,
    mesh: tl.constexpr,
):
    # Launch exactly one CTA on each rank. A separate kernel orders this
    # barrier after ALL preceding copy/consumer CTAs on the current stream.
    tle.distributed_barrier(
        mesh=mesh,
        device_dptr=dist_ctx,
        space="device",
        group_kind="block",
        barrier_kind="sync",
        order="acqrel",
        memory_scope="system",
        index=0,
        context_id=0,
    )


@dataclass
class AllGatherContext:
    output: torch.Tensor
    dist_ctx: object
    mesh: object
    world_size: int
    calls: int = 0
    last_config: dict[str, int] | None = None


def _all_gather_key(local_input, ctx):
    # Buffer addresses, values, call counts and the selected config are not
    # workload features: changing them must still reuse the tuned result.
    return to_hashable(local_input), to_hashable(ctx.output), ctx.world_size


@autotune(
    config_space=CONFIG_SPACE,
    key_fn=_all_gather_key,
    cache_version="tle-intra-push2d-barriers-v1",
)
def all_gather(
    local_input: torch.Tensor,
    ctx: AllGatherContext,
    *,
    block: int,
    num_warps: int,
    ctas_per_peer: int,
) -> torch.Tensor:
    """Gather contiguous, equally sized inputs into the registered output.

    Inputs must be separate from the output, with matching dtype/device.
    Every rank must invoke this in the same order on its context's stream.
    """
    elements_per_rank = local_input.numel()
    copy_grid = (ctx.world_size, min(ctas_per_peer, triton.cdiv(elements_per_rank, block)))
    if ctx.world_size > 1:
        # Fast ranks must not overwrite a slow rank's previous output while
        # it is still being consumed (or cleared before this invocation).
        _all_gather_barrier_kernel[(1, )](ctx.dist_ctx, ctx.mesh, num_warps=4)
    _all_gather_push_2d_kernel[copy_grid](
        local_input,
        ctx.output,
        ctx.dist_ctx,
        ctx.mesh,
        WORLD_SIZE=ctx.world_size,
        ELEM_PER_RANK=elements_per_rank,
        BLOCK=block,
        num_warps=num_warps,
    )
    if ctx.world_size > 1:
        _all_gather_barrier_kernel[(1, )](ctx.dist_ctx, ctx.mesh, num_warps=4)
    ctx.calls += 1
    ctx.last_config = {"block": block, "num_warps": num_warps, "ctas_per_peer": ctas_per_peer}
    return ctx.output


def _check_equal(actual, expected, group, label):
    # Agree collectively before raising: cleanup and the next call both need
    # every rank. Copy-only all-gather should be bitwise equal to torch.
    valid = torch.tensor([int(torch.equal(actual, expected))], dtype=torch.int32, device=actual.device)
    dist.all_reduce(valid, op=dist.ReduceOp.MIN, group=group)
    if valid.item() != 1:
        raise AssertionError(f"{label}: all-gather differs from torch on at least one rank")


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    mem_pool = tle.get_mem_pool()
    if mem_pool is None:
        raise RuntimeError("FlagCX memory pool is unavailable; check FlagCX build and environment variables.")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", str(world_size)))
    if world_size != local_world_size or rank != local_rank:
        raise RuntimeError("This tutorial requires a single node with rank == LOCAL_RANK")
    group = dist.group.WORLD

    M = int(os.environ.get("M", "8192"))
    N = int(os.environ.get("N", "12288"))
    if M <= 0 or N <= 0 or M % world_size:
        raise ValueError("M and N must be positive, and M must be divisible by world_size")
    m_per_rank = M // world_size
    dtype = torch.float16
    device = torch.device("cuda", local_rank)

    local_data = torch.randn((m_per_rank, N), dtype=dtype, device=device)

    with torch.cuda.use_mem_pool(mem_pool):
        ag_buffer = torch.empty((M, N), dtype=dtype, device=device)

    # The context holds the registered output and the FlagCX DevComm handles
    # used by remote pointers and the reusable intra-node barrier.
    dist_ctx = tle.create_dist_tensor(ag_buffer)
    ctx = AllGatherContext(
        output=ag_buffer,
        dist_ctx=dist_ctx,
        mesh=tle.device_mesh(tle.MeshConfig(device=world_size)),
        world_size=world_size,
    )
    golden = torch.empty_like(ag_buffer)
    for iteration in range(2):
        local_data.normal_().add_(rank + iteration * world_size)
        dist.all_gather_into_tensor(golden, local_data, group=group)
        ag_buffer.fill_(float("nan"))
        previous_calls = ctx.calls
        output = all_gather(local_data, ctx, autotune_pg=group)
        _check_equal(output, golden, group, f"Pass {iteration + 1}")
        if iteration == 1 and ctx.calls - previous_calls != 1:
            raise AssertionError("The second call should reuse the tuned configuration")
        if rank == 0:
            print(
                f"Pass {iteration + 1}: exact match; selected={ctx.last_config}; "
                f"operator invocations={ctx.calls - previous_calls}; shape={tuple(output.shape)}", flush=True)

    # Exercise the winner without the tuner's host-side collectives between
    # launches. Queue consumers on the same stream; the next entry barrier
    # must protect these reads even if a peer gets ahead. Keep just two
    # snapshots to bound memory use for the default 192 MiB output.
    selected = dict(ctx.last_config)
    snapshots = [torch.empty_like(ag_buffer) for _ in range(2)]
    for iteration, snapshot in enumerate(snapshots):
        local_data.fill_(rank + iteration * world_size)
        all_gather.fn(local_data, ctx, **selected)
        snapshot.copy_(ag_buffer)
    for iteration, snapshot in enumerate(snapshots):
        local_data.fill_(rank + iteration * world_size)
        dist.all_gather_into_tensor(golden, local_data, group=group)
        _check_equal(snapshot, golden, group, f"Back-to-back pass {iteration + 1}")
    if rank == 0:
        print("Back-to-back calls: exact match with changed inputs and output consumers", flush=True)

    torch.cuda.synchronize()
    # Cleanup is collective; let execution failures propagate to torchrun
    # instead of entering this from a rank-local finally block.
    tle.cleanup_communicator()


if __name__ == "__main__":
    main()
