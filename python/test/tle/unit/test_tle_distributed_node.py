"""Two-node TLE node-space remote-memory test.

Run this file through ``test_tle_distributed_node.sh`` on every node; do not
run it with plain ``python``. The test requires at least two nodes.

From the repository root, start one command on each node:

  # node 0
  NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
      bash python/test/tle/unit/test_tle_distributed_node.sh
  # node 1
  NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 \
      bash python/test/tle/unit/test_tle_distributed_node.sh

Use the same ``NNODES``, ``NPROC_PER_NODE``, ``MASTER_ADDR`` and
``MASTER_PORT`` on all nodes. ``MASTER_ADDR`` must be reachable from every
node. ``NPROC_PER_NODE`` selects the number of local GPUs (or ``gpu`` for all
visible GPUs). The shell script configures the required FlagCX environment.
The test prints PUT/GET PASS or FAILED for each rank and exits non-zero if
any rank fails.
"""

import os
import sys

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import triton.experimental.tle.language as tle


@triton.jit
def _node_put_kernel(
    buffer,
    src_offset,
    dst_offset,
    peer_world_rank,
    ctx: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    remote_dst = tle.remote(
        ctx,
        space="node",
        dtype=tl.float32,
        shard_id=peer_world_rank,
    )
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(buffer + src_offset + offsets)
    tl.store(remote_dst + dst_offset + offsets, values)


@triton.jit
def _node_get_kernel(
    buffer,
    src_offset,
    dst_offset,
    peer_world_rank,
    ctx: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    remote_src = tle.remote(
        ctx,
        space="node",
        dtype=tl.float32,
        shard_id=peer_world_rank,
    )
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(remote_src + src_offset + offsets)
    tl.store(buffer + dst_offset + offsets, values)


def main():
    mem_pool = tle.get_mem_pool()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    n_elements = 64
    get_bias = 10000

    local_world_size_text = os.environ.get("LOCAL_WORLD_SIZE")
    if local_world_size_text is None:
        raise RuntimeError("LOCAL_WORLD_SIZE is required; launch this test with torchrun")
    local_world_size = int(local_world_size_text)
    if local_world_size <= 0 or world_size % local_world_size != 0:
        raise ValueError(f"invalid topology: world_size={world_size}, "
                         f"LOCAL_WORLD_SIZE={local_world_size}")

    num_nodes = world_size // local_world_size
    if num_nodes < 2:
        raise RuntimeError(f"node remote test requires at least 2 nodes, got {num_nodes}")

    peer_world_rank = (rank + local_world_size) % world_size
    sender_world_rank = (rank - local_world_size) % world_size

    with torch.cuda.use_mem_pool(mem_pool):
        put_buffer = torch.empty(2 * n_elements, dtype=torch.float32, device="cuda")
        get_buffer = torch.empty(2 * n_elements, dtype=torch.float32, device="cuda")

        put_buffer[:n_elements] = (torch.arange(n_elements, dtype=torch.float32, device="cuda") + rank * 1000)
        put_buffer[n_elements:].fill_(-1)

        get_buffer[:n_elements] = (torch.arange(n_elements, dtype=torch.float32, device="cuda") + rank * 10000 +
                                   get_bias)
        get_buffer[n_elements:].fill_(-1)

    torch.cuda.synchronize()
    # Each communication path has its own buffer and registered context. The
    # collective registrations must happen in the same order on every rank.
    put_ctx = tle.create_dist_tensor(put_buffer)
    get_ctx = tle.create_dist_tensor(get_buffer)
    dist.barrier()

    # PUT: local put_buffer[0:N] -> peer put_buffer[N:2N].
    _node_put_kernel[(1, )](
        put_buffer,
        0,
        n_elements,
        peer_world_rank,
        ctx=put_ctx,
        BLOCK_SIZE=n_elements,
        num_ctas=1,
        num_warps=4,
    )
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()

    expected = torch.arange(n_elements, dtype=torch.float32, device="cuda") + sender_world_rank * 1000
    received = put_buffer[n_elements:]
    if torch.equal(received, expected):
        print(f"[Rank {rank}] node remote put from world rank {sender_world_rank}: PASSED")
    else:
        print(f"[Rank {rank}] node remote put from world rank {sender_world_rank}: FAILED")
        print(f"[Rank {rank}] actual[:4]={received[:4].tolist()}, expected[:4]={expected[:4].tolist()}")

    put_result = torch.tensor(
        [int(torch.equal(received, expected))],
        dtype=torch.int32,
        device="cuda",
    )
    dist.all_reduce(put_result, op=dist.ReduceOp.MIN)
    if not bool(put_result.item()):
        tle.cleanup_communicator()
        sys.exit(1)

    # GET: peer get_buffer[0:N] -> local get_buffer[N:2N].
    dist.barrier()
    _node_get_kernel[(1, )](
        get_buffer,
        0,
        n_elements,
        peer_world_rank,
        ctx=get_ctx,
        BLOCK_SIZE=n_elements,
        num_ctas=1,
        num_warps=4,
    )
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()

    expected = (torch.arange(n_elements, dtype=torch.float32, device="cuda") + peer_world_rank * 10000 + get_bias)
    received = get_buffer[n_elements:]
    get_validation_result = torch.all(received == expected)
    if bool(get_validation_result.item()):
        print(f"[Rank {rank}] node remote get from world rank {peer_world_rank}: PASSED")
    else:
        print(f"[Rank {rank}] node remote get from world rank {peer_world_rank}: FAILED")
        print(f"[Rank {rank}] actual[:4]={received[:4].tolist()}, expected[:4]={expected[:4].tolist()}")

    get_result = get_validation_result.to(dtype=torch.int32).reshape(1)
    dist.all_reduce(get_result, op=dist.ReduceOp.MIN)

    tle.cleanup_communicator()
    if not bool(get_result.item()):
        sys.exit(1)


if __name__ == "__main__":
    main()
