# Copyright 2026- Xcoresigma Technology Co., Ltd

import torch
import torch_npu  # noqa: F401  (loads libtorch_npu before the NPU driver utils)
import torch.distributed as dist
import triton
import triton.language as tl
import triton.experimental.tle as tle

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

N = 64  # block size per rank (power of two for tl.arange)
F32_SIZE = tl.constexpr(4)  # sizeof(tl.float32) in bytes — size arg of the RMA calls


@triton.jit
def kernel_put(put_buf_ptr, src_ptr, rank, peer, n: tl.constexpr):
    """PUT: one-sided copy of the local block (src_ptr)."""
    dst = put_buf_ptr + rank * n  # symmetric address of our segment
    tle.dsa.ascend.shmem_put_nbi(src_ptr, dst, n * F32_SIZE, peer)
    tle.shmem_fence()


@triton.jit
def kernel_get(get_src_ptr, recv_ptr, peer, n: tl.constexpr):
    """GET: one-sided copy of the PEER's segment of get_src."""
    src = get_src_ptr + peer * n  # symmetric address of the peer's segment
    tle.dsa.ascend.shmem_get_nbi(recv_ptr, src, n * F32_SIZE, peer)
    tle.shmem_fence()


def main():
    tle.init_communicator()
    pe = dist.get_rank()
    world = dist.get_world_size()
    peer = (pe + 1) % world
    print(f"[Rank {pe}] init: world={world} peer={peer}", flush=True)

    # Symmetric buffers; rank r owns the segment [r*N, (r+1)*N).
    put_buf = tle.create_dist_tensor(torch.zeros(N * world, dtype=torch.float32, device="npu"))
    get_src = tle.create_dist_tensor(torch.zeros(N * world, dtype=torch.float32, device="npu"))
    src_local = torch.arange(N, dtype=torch.float32, device="npu") + pe * 1000
    recv = torch.zeros(N, dtype=torch.float32, device="npu")

    get_src[pe * N:(pe + 1) * N] = (torch.arange(N, dtype=torch.float32, device="npu") + pe * 1000 + 500)
    dist.barrier()

    # PUT our block into our segment of the PEER's put_buf copy; GET the
    # peer's get_src segment into our recv buffer.
    kernel_put[(1, )](put_buf, src_local, pe, peer, N)
    kernel_get[(1, )](get_src, recv, peer, N)
    torch.npu.synchronize()
    dist.barrier()

    fails = []

    # 1) PUT
    prev = (pe - 1 + world) % world
    got_put = put_buf.cpu().tolist()[prev * N:(prev + 1) * N]
    exp_put = (torch.arange(N, dtype=torch.float32, device="npu") + prev * 1000).cpu().tolist()
    ok = got_put == exp_put
    print(
        f"[Rank {pe}] put: put_buf[seg {prev}]={got_put[:3]}... "
        f"(expect rank {prev}'s block) {'OK' if ok else 'FAIL'}", flush=True)
    if not ok:
        fails.append("put")

    # 2) GET
    exp_get = (torch.arange(N, dtype=torch.float32, device="npu") + peer * 1000 + 500).cpu().tolist()
    got_get = recv.cpu().tolist()
    ok = got_get == exp_get
    print(f"[Rank {pe}] get: recv matches peer seg "
          f"{'OK' if ok else 'FAIL'}", flush=True)
    if not ok:
        fails.append("get")

    status = torch.tensor([0 if fails else 1], dtype=torch.int32, device="npu")
    all_ok = [torch.zeros(1, dtype=torch.int32, device="npu") for _ in range(world)]
    dist.all_gather(all_ok, status)
    if any(s.item() == 0 for s in all_ok):
        raise AssertionError(f"case 14 failed on some rank: {fails}")
    if pe == 0:
        print(f"{GREEN}[SUCCESS]{RESET} shmem_put_nbi / shmem_get_nbi / "
              f"shmem_fence verified on {world} ranks!", flush=True)

    tle.cleanup_communicator(put_buf)
    tle.cleanup_communicator(get_src)


# cmd: torchrun --nproc-per-node=2 python/tutorials/tle/dsa/13-shmem-put-get-nbi.py
if __name__ == "__main__":
    main()
