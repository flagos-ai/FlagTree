#!/usr/bin/env python3
"""Two-node TLE all-gather tutorial using a push-2d topology.

The topology is fixed to two nodes with four GPUs per node.  Every rank owns
16 MiB of float32 input.  One WORLD_SIZE-CTA kernel overlaps three roles:

1. push the local segment to the same local rank on the other node;
2. relay local and newly arrived remote segments to the other local GPUs;
3. wait until all remote source segments are ready.

The tutorial runs both torch.distributed and TLE once, then compares their
complete all-gather outputs element by element.
"""

import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

ELEMENTS_PER_RANK = 4 * 1024 * 1024  # 16 MiB of float32 per rank.
BLOCK = 4096
NODE_BLOCK = 1 * 1024 * 1024  # 4 MiB of float32 per node PUT.
NUM_WARPS = 32


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
            # The self CTA is the completion CTA.  The local segment requires
            # no signal because it was staged before this kernel was launched.
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


def push_2d_all_gather(
    local_input: torch.Tensor,
    registered_output: torch.Tensor,
    device_dptr,
    rank: int,
    local_rank: int,
    node_rank: int,
    world_size: int,
    local_world_size: int,
) -> torch.Tensor:
    """Stage one local input and launch one result-ready all-gather kernel."""
    segment_begin = rank * ELEMENTS_PER_RANK
    registered_output[segment_begin:segment_begin + ELEMENTS_PER_RANK].copy_(local_input)

    push_2d_all_gather_kernel[(world_size, )](
        registered_output,
        device_dptr,
        local_rank,
        node_rank,
        1,
        LOCAL_WORLD_SIZE=local_world_size,
        WORLD_SIZE=world_size,
        ELEM_PER_RANK=ELEMENTS_PER_RANK,
        NUM_NODE_CHUNKS=triton.cdiv(ELEMENTS_PER_RANK, NODE_BLOCK),
        NUM_LOCAL_CHUNKS=triton.cdiv(ELEMENTS_PER_RANK, BLOCK),
        NODE_BLOCK=NODE_BLOCK,
        BLOCK=BLOCK,
        num_warps=NUM_WARPS,
    )
    return registered_output


def main() -> None:
    mem_pool = tle.get_mem_pool()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    if local_world_size <= 0 or world_size % local_world_size:
        raise RuntimeError(f"invalid topology: world_size={world_size}, "
                           f"local_world_size={local_world_size}")
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

    try:
        dist.all_gather_into_tensor(torch_reference, local_input)
        torch.cuda.synchronize()

        output = push_2d_all_gather(
            local_input,
            registered_output,
            device_dptr,
            rank,
            local_rank,
            node_rank,
            world_size,
            local_world_size,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(output, torch_reference, atol=0, rtol=0)

        if rank == 0:
            print(
                f"TLE push-2d all-gather matches torch: output shape={tuple(output.shape)}",
                flush=True,
            )
    finally:
        tle.cleanup_communicator()


if __name__ == "__main__":
    main()
