"""Function-level autotuning for push-2d TLE all-gather on sixteen GPUs across two nodes.

Run the corresponding command on each node, using the reachable address of node 0:

    NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash python/tutorials/tle/06-inter-node-allgather.sh
    NODE_RANK=1 MASTER_ADDR=10.0.0.1 bash python/tutorials/tle/06-inter-node-allgather.sh

The launcher starts eight worker processes per node (sixteen in total), with
one GPU per process.  Each node uses the first eight GPUs in its
CUDA_VISIBLE_DEVICES list, even if more GPUs are visible.  The Python code
requires WORLD_SIZE=16 and LOCAL_WORLD_SIZE=8 and uses a 2-by-8 device mesh.
Signals use sixteen slots, indexed by the source's global rank.  Each
receiver waits for one signal per epoch from every other source; its own
input is staged locally and needs no signal.

Each rank has 16 MiB of float32 input.  Four candidates combine BLOCK values
of 2048/4096 with NODE_BLOCK values of 524288/1048576, measured in elements.
The all-gather kernel uses NUM_WARPS=32 for every candidate.  Every rank must
call the decorated function with autotune_pg=dist.group.WORLD, matching the
FlagCX world communicator.

Every all-gather execution, including warmup, measurement, and cached
execution, advances its signal epoch and ends with a GPU world barrier
before the communication buffer can be reused.  The barrier and local input
copy are included in the measured operator latency.  One context tracks the
epochs of one registered buffer; do not reset its epoch or invoke the
all-gather concurrently on multiple streams using that context.

The example calls the decorated function twice with different input values
and checks exact equality with PyTorch's all-gather result after each call.
On the first call, group rank 0 checks for a matching disk cache entry;
all ranks tune together only on a cache miss.  The second call reuses the
in-memory result.  Set FLAGTREE_AUTOTUNE_CACHE_DIR to choose the disk cache
directory.  Bump cache_version below if the helper kernels or communication
protocol change.
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

ELEMENTS_PER_RANK = 4 * 1024 * 1024  # 16 MiB of float32 per rank.
NUM_WARPS = 32
CONFIG_SPACE = [
    {"block": block, "node_block": node_block}
    for block in (2048, 4096)
    for node_block in (512 * 1024, 1024 * 1024)
]


@triton.jit(do_not_specialize=["local_rank", "node_rank", "epoch"])
def push_2d_all_gather_kernel(
    registered_output,
    device_dptr: tl.constexpr,
    local_rank,
    node_rank,
    epoch,
    LOCAL_WORLD_SIZE: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    ELEM_PER_RANK: tl.constexpr,
    NUM_NODE_CHUNKS: tl.constexpr,
    NUM_LOCAL_CHUNKS: tl.constexpr,
    NODE_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Push one complete all-gather segment per producer CTA."""
    peer_rank = tl.program_id(0)
    global_rank = node_rank * LOCAL_WORLD_SIZE + local_rank
    peer_node = peer_rank // LOCAL_WORLD_SIZE
    peer_local_rank = peer_rank % LOCAL_WORLD_SIZE

    if peer_local_rank == local_rank:
        if peer_rank != global_rank:
            # Inter-node producer.  The local segment already lives in its
            # final all-gather position inside the registered output buffer.
            segment_offset = global_rank * ELEM_PER_RANK
            node_offsets = tl.arange(0, NODE_BLOCK)
            for chunk_id in tl.range(0, NUM_NODE_CHUNKS):
                block_offset = chunk_id * NODE_BLOCK
                valid_n = tl.minimum(ELEM_PER_RANK - block_offset, NODE_BLOCK)
                mask = node_offsets < valid_n
                remote_dst = tle.remote(
                    device_dptr,
                    shard_id=peer_rank,
                    space="node",
                    dtype=tl.float32,
                    context_id=0,
                    coopkind="block",
                )
                values = tl.load(
                    registered_output + segment_offset + block_offset + node_offsets,
                    mask=mask,
                )
                tl.store(
                    remote_dst + segment_offset + block_offset + node_offsets,
                    values,
                    mask=mask,
                )

            # Context 0 orders this signal behind the node PUT operations.
            tle.signal(
                device_dptr,
                peer_rank,
                slot_id=global_rank,
                op="inc",
                space="world",
                group_kind="block",
                context_id=0,
            )
        else:
            # The self CTA waits for every other source's segment.  Its own
            # input was staged before launch and does not produce a signal.
            for source_rank in tl.static_range(0, WORLD_SIZE):
                if source_rank != global_rank:
                    tle.signal_wait(
                        device_dptr,
                        slot_id=source_rank,
                        wait_kind="signal",
                        target=epoch,
                        group_kind="block",
                        context_id=0,
                    )
    else:
        # Relay the source with this CTA's peer_node and the current local
        # rank to the local destination selected by peer_local_rank.
        source_rank = peer_node * LOCAL_WORLD_SIZE + local_rank
        segment_offset = source_rank * ELEM_PER_RANK

        if peer_node != node_rank:
            tle.signal_wait(
                device_dptr,
                slot_id=source_rank,
                wait_kind="signal",
                target=epoch,
                group_kind="block",
                context_id=0,
            )

        # Resolve the destination GPU pointer once, then copy the complete
        # segment through a loop of BLOCK-element local tiles.
        remote_dst = tle.remote(
            device_dptr,
            shard_id=peer_local_rank,
            space="device",
            dtype=tl.float32,
            offset=segment_offset,
        )
        local_offsets = tl.arange(0, BLOCK)
        for chunk_id in tl.range(0, NUM_LOCAL_CHUNKS):
            block_offset = chunk_id * BLOCK
            mask = block_offset + local_offsets < ELEM_PER_RANK
            values = tl.load(
                registered_output + segment_offset + block_offset + local_offsets,
                mask=mask,
            )
            tl.store(remote_dst + block_offset + local_offsets, values, mask=mask)

        tle.signal(
            device_dptr,
            peer_local_rank,
            slot_id=source_rank,
            op="inc",
            space="intra_node",
            group_kind="block",
            context_id=0,
        )


@triton.jit
def _finish_all_gather_epoch(device_dptr: tl.constexpr, mesh: tl.constexpr):
    # A separate one-CTA kernel runs after every CTA in the all-gather kernel
    # has completed.  All ranks must reach it before any rank reuses its buffer.
    tle.distributed_barrier(
        mesh=mesh,
        device_dptr=device_dptr,
        space="world",
        group_kind="block",
        barrier_kind="sync",
        order="acqrel",
        index=0,
    )


@dataclass
class AllGatherContext:
    registered_output: torch.Tensor
    device_dptr: object
    rank: int
    local_rank: int
    node_rank: int
    world_size: int
    local_world_size: int
    world_mesh: object
    epoch: int = 0
    last_config: dict[str, int] | None = None


def _all_gather_key(local_input, ctx, **kwargs):
    # Epochs, buffer addresses, input values, and the last selected config must
    # not produce a new workload key on every benchmark or cached invocation.
    return (to_hashable(local_input), to_hashable(ctx.registered_output), ctx.rank,
            ctx.world_size, ctx.local_world_size, NUM_WARPS)


@autotune(config_space=CONFIG_SPACE, key_fn=_all_gather_key, cache_version="tle-push2d-global-slots-v1")
def push_2d_all_gather(
    local_input: torch.Tensor,
    ctx: AllGatherContext,
    *,
    block: int,
    node_block: int,
) -> torch.Tensor:
    """Run one repeatable all-gather; all ranks must use the same candidate."""
    ctx.epoch += 1
    elements_per_rank = local_input.numel()
    registered_output = ctx.registered_output
    segment_begin = ctx.rank * elements_per_rank
    registered_output[segment_begin:segment_begin + elements_per_rank].copy_(local_input)

    push_2d_all_gather_kernel[(ctx.world_size, )](
        registered_output,
        ctx.device_dptr,
        ctx.local_rank,
        ctx.node_rank,
        ctx.epoch,
        LOCAL_WORLD_SIZE=ctx.local_world_size,
        WORLD_SIZE=ctx.world_size,
        ELEM_PER_RANK=elements_per_rank,
        NUM_NODE_CHUNKS=triton.cdiv(elements_per_rank, node_block),
        NUM_LOCAL_CHUNKS=triton.cdiv(elements_per_rank, block),
        NODE_BLOCK=node_block,
        BLOCK=block,
        num_warps=NUM_WARPS,
    )
    # The next invocation's copy is queued behind this barrier on the same
    # stream, including during the autotuner's back-to-back warmup/repeat loops.
    _finish_all_gather_epoch[(1, )](ctx.device_dptr, ctx.world_mesh, num_warps=4)
    ctx.last_config = {"block": block, "node_block": node_block}
    return registered_output


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # TLE initializes the NCCL and FlagCX communicators here.
    mem_pool = tle.get_mem_pool()
    if mem_pool is None:
        raise RuntimeError("FlagCX memory pool is unavailable; check the FlagCX build and environment.")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    group = dist.group.WORLD

    # Match the launcher's two nodes with eight GPUs each.  create_dist_tensor
    # provides sixteen user signal slots, one for each global source rank.
    if (world_size, local_world_size) != (16, 8):
        raise RuntimeError("This tutorial requires two nodes with eight GPUs per node; "
                           f"got world_size={world_size}, local_world_size={local_world_size}")
    node_rank = rank // local_world_size
    expected_rank = node_rank * local_world_size + local_rank
    if rank != expected_rank:
        raise RuntimeError(f"rank mapping mismatch: rank={rank}, node_rank={node_rank}, "
                           f"local_rank={local_rank}")

    local_input = torch.full(
        (ELEMENTS_PER_RANK, ),
        float(rank),
        dtype=torch.float32,
        device="cuda",
    )

    # Every rank allocates an equally sized symmetric all-gather output.
    with torch.cuda.use_mem_pool(mem_pool):
        registered_output = torch.empty(
            world_size * ELEMENTS_PER_RANK,
            dtype=torch.float32,
            device="cuda",
        )

    device_dptr = tle.create_dist_tensor(registered_output)
    torch_reference = torch.empty_like(registered_output)
    ctx = AllGatherContext(
        registered_output=registered_output,
        device_dptr=device_dptr,
        rank=rank,
        local_rank=local_rank,
        node_rank=node_rank,
        world_size=world_size,
        local_world_size=local_world_size,
        world_mesh=tle.device_mesh(tle.MeshConfig(node=2, device=8)),
    )

    for iteration in range(2):
        # Change values between the first and cached calls so stale data
        # from an earlier epoch cannot make the correctness check pass.
        local_input.fill_(float(rank + iteration * world_size))
        dist.all_gather_into_tensor(torch_reference, local_input, group=group)
        registered_output.fill_(float("nan"))
        torch.cuda.synchronize()
        dist.barrier(group=group)

        previous_epoch = ctx.epoch
        output = push_2d_all_gather(local_input, ctx, autotune_pg=group)
        torch.cuda.synchronize()

        # All ranks agree on validation before entering another invocation
        # or the collective communicator cleanup.
        valid = torch.tensor(
            [int(torch.equal(output, torch_reference))], dtype=torch.int32, device=output.device,
        )
        dist.all_reduce(valid, op=dist.ReduceOp.MIN, group=group)
        if valid.item() != 1:
            raise AssertionError(f"All-gather differs from torch on at least one rank, iteration={iteration}")
        if rank == 0:
            print(
                f"Pass {iteration + 1}: TLE matches torch; selected={ctx.last_config}; "
                f"operator invocations={ctx.epoch - previous_epoch}; output shape={tuple(output.shape)}",
                flush=True,
            )

    # Cleanup contains a collective barrier.  A rank-local execution failure
    # must propagate to the launcher instead of waiting for peers in a finally.
    tle.cleanup_communicator()


if __name__ == "__main__":
    main()
