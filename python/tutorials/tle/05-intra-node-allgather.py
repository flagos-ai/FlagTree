"""
Intra-node AllGather with FlagTree TLE device remote pointers.
This tutorial implements a single-node all-gather operator using
FlagTree TLE.
Run with a FlagTree environment, for example:

    export FLAGCX_MEM_ENABLE=1
    export FLAGCX_USE_HETERO_COMM=1
    export FLAGCX_VMM_ENABLE=0
    export FLAGCX_P2P_DISABLE=1
    export CUDA_VISIBLE_DEVICES=0,1
    # Optional: set FLAGCX_IB_HCA to the HCA list for your machine.
    torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/test_tle_intra_node_allgather.py"

If you explicitly disabled distributed support with USE_FLAGCX=0, USE_DIST=0,
or USE_TLE_DIST=0, it might be necessary to reset these settings before running this tutorial.
"""

import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def _all_gather_push_2d_kernel(
    local_ptr,
    ag_ptr,
    dist_ctx: tl.constexpr,
    mesh: tl.constexpr,
    ELEM_PER_RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    peer = tl.program_id(0)
    block_id = tl.program_id(1)
    local_rank = tle.shard_id(mesh, "device", device_dptr=dist_ctx)
    dst_base = local_rank * ELEM_PER_RANK
    offsets = block_id * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < ELEM_PER_RANK
    vals = tl.load(local_ptr + offsets, mask=mask, other=0.0)

    if peer != local_rank:
        dst_ptr = tle.remote(
            dist_ctx,
            shard_id=peer,
            space="device",
            dtype=ag_ptr.dtype.element_ty,
            offset=dst_base,
        )
        tl.store(dst_ptr + offsets, vals, mask=mask)
    else:
        tl.store(ag_ptr + dst_base + offsets, vals, mask=mask)


@triton.jit
def _all_gather_signal_kernel(
    dist_ctx: tl.constexpr,
    mesh: tl.constexpr,
):
    peer = tl.program_id(0)
    local_rank = tle.shard_id(mesh, "device", device_dptr=dist_ctx)

    if peer != local_rank:
        # Every remote rank contributes one completion to slot 0. The paired
        # signal_wait below waits for all remote contributors.
        tle.signal(
            dist_ctx,
            peer,
            slot_id=0,
            op="inc",
            space="intra_node",
            group_kind="block",
            context_idx=0,
        )


@triton.jit(do_not_specialize=["signal_target"])
def _all_gather_wait_kernel(
    dist_ctx: tl.constexpr,
    signal_target,
):
    """Wait until every remote shard in this rank's output is ready."""
    tle.signal_wait(
        dist_ctx,
        slot_id=0,
        wait_kind="signal",
        target=signal_target,
        group_kind="block",
        context_idx=0,
    )


def _rank_print(rank: int, *items):
    dist.barrier()
    for cur_rank in range(dist.get_world_size()):
        if cur_rank == rank:
            print(*items, flush=True)
        dist.barrier()


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    mem_pool = tle.get_mem_pool()
    if mem_pool is None:
        raise RuntimeError("FlagCX memory pool is unavailable; check FlagCX build and environment variables.")

    rank = dist.get_rank()  # Obtain rank and world_size
    world_size = dist.get_world_size()
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", str(world_size)))
    assert world_size == local_world_size, "This tutorial is designed for a single node"

    M = 8192
    N = 12288
    assert M % world_size == 0
    m_per_rank = M // world_size
    dtype = torch.float16
    device = torch.device("cuda", local_rank)

    local_data = torch.randn((m_per_rank, N), dtype=dtype, device=device)

    with torch.cuda.use_mem_pool(mem_pool):
        ag_buffer = torch.empty((M, N), dtype=dtype, device=device)

    # dist_ctx owns the registered all-gather buffer plus the FlagCX DevComm
    # handles used by tle.remote, tle.signal, and tle.signal_wait.
    dist_ctx = tle.create_dist_tensor(ag_buffer)

    golden = torch.empty((M, N), dtype=dtype, device=device)
    dist.all_gather_into_tensor(golden, local_data)

    ag_buffer.fill_(-1)

    torch.cuda.synchronize()
    dist.barrier()

    elem_per_rank = m_per_rank * N
    block = 1024
    num_blocks = triton.cdiv(elem_per_rank, block)

    # 2D copy grid: split each peer transfer into independent chunks.
    # The signal is written by a second kernel so it is ordered after all copy chunks in this stream.
    copy_grid = (world_size, num_blocks)
    signal_grid = (world_size, )
    wait_grid = (1, )
    mesh = tle.device_mesh(tle.MeshConfig(device=world_size))
    signal_target = world_size - 1

    def launch_tle_all_gather():
        _all_gather_push_2d_kernel[copy_grid](
            local_data,
            ag_buffer,
            dist_ctx,
            mesh,
            ELEM_PER_RANK=elem_per_rank,
            BLOCK=block,
            num_warps=4,
        )
        _all_gather_signal_kernel[signal_grid](
            dist_ctx,
            mesh,
            num_warps=4,
        )
        _all_gather_wait_kernel[wait_grid](
            dist_ctx,
            signal_target,
            num_warps=1,
        )

    launch_tle_all_gather()
    torch.cuda.synchronize()
    dist.barrier()

    _rank_print(rank, f"Rank {rank} FlagTree Result:", ag_buffer)
    assert torch.allclose(golden, ag_buffer, atol=1e-5, rtol=1e-5)
    _rank_print(rank, f"Rank {rank} Pass!")

    tle.cleanup_communicator()


if __name__ == "__main__":
    main()
