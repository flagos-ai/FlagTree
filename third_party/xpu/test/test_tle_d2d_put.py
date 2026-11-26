import torch
import torch.distributed as dist
import triton
import triton.language as tl

import triton.experimental.tle.language as tle


@triton.jit
def lsa_put_kernel(input_ptr, device_dptr, MY_RANK: tl.constexpr):
    pid = tl.program_id(0)
    peer = (MY_RANK + 1) % tle.n_pes(device_dptr)
    remote_mem = tle.remote(device_dptr, space="device", dtype=tl.float32, shard_id=peer, offset=pid)
    value = tl.load(input_ptr + pid)
    if tl.cast(remote_mem, tl.uint64) != 0:
        tl.store(remote_mem, value)


def main():
    mem_pool = tle.get_mem_pool()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        raise ValueError("D2D put requires at least two ranks")

    n = 64
    values = torch.arange(n, dtype=torch.float32)
    with torch.cuda.use_mem_pool(mem_pool):
        receive = torch.full((n,), -1.0, dtype=torch.float32, device="cpu").to("cuda")
    source = (values + rank * 1000).to("cuda")
    device_dptr = tle.create_dist_tensor(receive)
    try:
        lsa_put_kernel[(n,)](source, device_dptr, rank, num_ctas=1, num_warps=4)
        torch.cuda.synchronize()
        dist.barrier()

        sender = (rank - 1) % world_size
        actual = receive.cpu()
        torch.testing.assert_close(actual, values + sender * 1000, rtol=0, atol=0)
        print(f"[Rank {rank}] PASSED: received {n} floats from rank {sender}, sample={actual[:4].tolist()}", flush=True)
    finally:
        tle.cleanup_communicator()


if __name__ == "__main__":
    main()
