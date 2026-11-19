# Copyright 2026- Xcoresigma Technology Co., Ltd

import torch
import torch_npu
import torch.distributed as dist
import triton
import triton.language as tl
import triton.experimental.tle as tle

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

# signal_wait needs an int64/uint64 signal buffer.
# aclshmem_wait walks signal slots 64 bytes apart: 64 / sizeof(int64) = 8 elements.
SIGNAL_SLOT_STRIDE = 8

N = 64  # block size, must be a power of two for tl.arange


@triton.jit
def kernel_notify_wait_consume(
    in_ptr,
    out_ptr,  # output block, receives the neighbour's data
    peer_mem_ptr,  # symmetric data buffer: [rank_size, N]
    signal_mem_ptr,  # symmetric signal buffer
    rank,
    rank_size,
    N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    peer_rank = (rank + 1) % rank_size
    src_rank = (rank + rank_size - 1) % rank_size
    offs = tl.arange(0, N)

    if pid == 0:
        # Producer
        data = tl.load(in_ptr + offs)

        peer_data_ptr = tle.remote(peer_mem_ptr, space="device", dtype=tl.float32, shard_id=peer_rank)

        dst_offs = rank * N + offs
        tl.store(peer_data_ptr + dst_offs, data)

        tle.shmem_fence()
        tle.signal(signal_mem_ptr, peer_rank, value=1, op="set")

    if pid == 1:
        # Consumer
        token = tle.signal_wait(signal_mem_ptr, target=1, scope="rank")

        peer_mem_ptr = tle.dsa.ascend.consume_token(peer_mem_ptr, token)
        data = tl.load(peer_mem_ptr + src_rank * N + offs)
        tl.store(out_ptr + offs, data)


def notify_wait_consume(A, out, peer_mem, signal_mem, rank, rank_size):
    """Launcher wrapper function (one producer + one consumer program)."""
    kernel_notify_wait_consume[2, 1, 1](
        A,
        out,
        peer_mem,
        signal_mem,
        rank,
        rank_size,
        N=N,
    )


def run_test_distributed():
    tle.init_communicator()
    pe = dist.get_rank()
    world_size = dist.get_world_size()

    dtype = torch.float32

    # Symmetric data buffer (one N-element section per rank) and a symmetric
    # int64 signal buffer holding a single 64B slot.
    peer_mem = tle.create_dist_tensor(torch.empty([N * world_size], dtype=dtype, device="npu"))
    signal_mem = tle.create_dist_tensor(torch.empty([SIGNAL_SLOT_STRIDE], dtype=torch.int64, device="npu"))

    # Local block written by this rank: rank * 1000 + arange(N).
    A = (torch.arange(N, dtype=dtype, device="npu") + pe * 1000).clone()
    out = torch.zeros([N], dtype=dtype, device="npu")

    signal_mem.fill_(0)
    peer_mem.zero_()
    dist.barrier()

    notify_wait_consume(A, out, peer_mem, signal_mem, pe, world_size)
    torch.npu.synchronize()

    # src * 1000 + arange(N), where src = (rank - 1 + world_size) % world_size.
    src_rank = (pe - 1 + world_size) % world_size
    expected = torch.arange(N, dtype=dtype, device="npu") + src_rank * 1000

    passed = torch.tensor([1], dtype=torch.int32).npu()
    error_msg = ""
    try:
        torch.testing.assert_close(expected, out, rtol=1e-5, atol=1e-5)
    except AssertionError as e:
        passed[0] = 0
        error_msg = str(e)

    all_passed = [torch.zeros(1, dtype=torch.int32).npu() for _ in range(world_size)]
    dist.all_gather(all_passed, passed)

    dist.barrier()
    for rank_id in range(world_size):
        if pe == rank_id:
            if all_passed[rank_id].item() == 1:
                print(
                    f"{GREEN}[PASS]{RESET} Rank {pe}: received rank {src_rank}'s block "
                    f"(src * 1000 + arange) via notify/wait/consume_token.",
                    flush=True,
                )
            else:
                print(
                    f"{RED}[FAIL]{RESET} Rank {pe}: out does not match rank {src_rank}'s "
                    f"block. Details:\n{error_msg}",
                    flush=True,
                )
        dist.barrier()

    # Release the symmetric memory resources (finalize on last cleanup).
    tle.cleanup_communicator(peer_mem)
    tle.cleanup_communicator(signal_mem)

    if any(r.item() == 0 for r in all_passed):
        raise AssertionError(f"notify/wait/consume_token failed on rank {pe}")

    if pe == 0:
        print(f"\n{GREEN}[SUCCESS]{RESET} notify + wait + consume_token ring push "
              f"verified on {world_size} ranks!")


# cmd: torchrun --nproc-per-node=2 python/tutorials/tle/dsa/13-ascend-notify-wait-consume-token.py
if __name__ == "__main__":
    run_test_distributed()
