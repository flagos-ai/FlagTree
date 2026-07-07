# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import triton.experimental.tle.language as tle


@triton.jit
def lsa_read_kernel(
    dev_mem_ptr,
    output_ptr,
    N: tl.constexpr,
    MY_RANK: tl.constexpr,
    N_RANKS: tl.constexpr,
):
    pid = tl.program_id(0)
    peer = (MY_RANK + 1) % N_RANKS

    remote_mem = tle.remote(
        dev_mem_ptr,
        space="device",
        dtype=tl.float32,
        shard_id=peer,
        offset=pid,
    )
    val = tl.load(remote_mem)
    tl.store(output_ptr + pid, val)


def main():
    mem_pool = tle.get_mem_pool()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    print(f"[Rank {rank}] Starting D2D test (world_size={world_size})")

    N = 64

    with torch.cuda.use_mem_pool(mem_pool):
        # torch.arange may return only 512-byte aligned memory, but FlagCX
        # symmetric window / flagcxGetIntraPointerC requires 4KB page aligned
        # buffers. Clone to force reallocation with proper alignment.
        buf_tensor = (torch.arange(N, dtype=torch.float32, device="cuda") + rank * 1000).clone()

    print(f"[Rank {rank}] buf_tensor info: "
          f"data_ptr={buf_tensor.data_ptr():#x}, "
          f"is_contiguous={buf_tensor.is_contiguous()}, "
          f"storage_offset={buf_tensor.storage_offset()}, "
          f"stride={buf_tensor.stride()}, "
          f"sample={buf_tensor[:4].tolist()}")

    dev_comm_ptr, dev_mem_ptr = tle.create_comm_tensor(buf_tensor)

    print(f"[Rank {rank}] dev_comm_ptr={dev_comm_ptr:#x}, dev_mem_ptr={dev_mem_ptr:#x}")

    output = torch.zeros(N, dtype=torch.float32, device="cuda")

    dist.barrier()

    grid = (N, )
    lsa_read_kernel[grid](
        dev_mem_ptr,
        output,
        N=N,
        MY_RANK=rank,
        N_RANKS=world_size,
    )
    torch.cuda.synchronize()

    import sys
    peer_rank = (rank + 1) % world_size
    expected = torch.arange(N, dtype=torch.float32, device="cuda") + peer_rank * 1000
    if torch.allclose(output, expected):
        print(f"[Rank {rank}] PASSED: read peer rank {peer_rank}")
        print(f"[Rank {rank}] sample output[:4] = {output[:4].tolist()}")
    else:
        print(f"[Rank {rank}] FAILED: read peer rank {peer_rank}")
        print(f"[Rank {rank}] expected[:4] = {expected[:4].tolist()}")
        print(f"[Rank {rank}] output[:4] = {output[:4].tolist()}")
        sys.exit(1)

    tle.cleanup_communicator()
    print(f"[Rank {rank}] Done")


if __name__ == "__main__":
    main()
