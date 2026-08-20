"""Multi-device check that tle.remote(space="device") really reads the peer.

The compile-only coverage lives in test_tle_remote_pointers.py. It can only
prove that flagcxGetIntraPointerC is called with the right ABI; a lowering that
returned the local pointer, or that passed an element index where the ABI wants
a byte offset, would pass all of it. This file runs the read on two devices and
checks the values come from the peer rank at the right element.

Trunk's python/test/tle/unit/test_tle_distributed_d2d.py covers the constant-peer
read and iluvatar CI runs it directly; this file adds the runtime-peer shard id,
which reaches the FlagCX call as a register instead of a folded constant.

Launch it with torchrun on two devices (see third_party/iluvatar/test_triton.sh);
under a plain pytest run it skips, so it stays part of the single-device suite
without needing a distributed setup.
"""
import os

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle  # pyright: ignore[reportMissingImports]

N = 64
RANK_STRIDE = 1000


# No in-kernel device barrier here. Both ranks fill their buffer before the
# host-side dist.barrier(), so the peer data is already visible, and the FlagCX
# intra barrier keys on a barrier_index slot: with one grid of N blocks every
# CTA would arrive on slot 0, letting one rank bump the epoch while a peer CTA
# still waits on the previous one. The barrier has its own two-device coverage
# in test_tle_distributed_barrier_multi_gpu.py, on a single-block grid.
@triton.jit
def _remote_read_kernel(out_ptr, device_dptr: tl.constexpr, MY_RANK: tl.constexpr, N_RANKS: tl.constexpr):
    pid = tl.program_id(0)
    peer = (MY_RANK + 1) % N_RANKS

    remote_mem = tle.remote(
        device_dptr,
        space="device",
        dtype=tl.float32,
        shard_id=peer,
        offset=pid,
    )
    val = tl.load(remote_mem)
    tl.store(out_ptr + pid, val)


@triton.jit
def _remote_read_runtime_peer_kernel(out_ptr, device_dptr: tl.constexpr, mesh: tl.constexpr):
    pid = tl.program_id(0)
    local_rank = tle.shard_id(mesh, 'device', device_dptr=device_dptr)
    peer = (local_rank + 1) % mesh.shape[0]

    remote_mem = tle.remote(
        device_dptr,
        space="device",
        dtype=tl.float32,
        shard_id=peer,
        offset=pid,
    )
    tl.store(out_ptr + pid, tl.load(remote_mem))


def _setup():
    import torch.distributed as dist

    # tle.get_mem_pool() initializes the process group, so it has to run before
    # anything asks for the rank.
    mem_pool = tle.get_mem_pool()
    rank = dist.get_rank()
    with torch.cuda.use_mem_pool(mem_pool):
        # torch.arange may return only 512-byte aligned memory, but the FlagCX
        # symmetric window / flagcxGetIntraPointerC requires 4KB page aligned
        # buffers. Clone to force reallocation with proper alignment.
        buf = (torch.arange(N, dtype=torch.float32, device="cuda") + rank * RANK_STRIDE).clone()
    return tle.create_dist_tensor(buf)


def _expected(peer_rank):
    return torch.arange(N, dtype=torch.float32, device="cuda") + peer_rank * RANK_STRIDE


def _run_remote_read_check():
    import torch.distributed as dist

    device_dptr = _setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    peer_rank = (rank + 1) % world_size
    expected = _expected(peer_rank)

    # Compile-time peer: the shard id folds into the FlagCX call.
    out = torch.zeros(N, dtype=torch.float32, device="cuda")
    dist.barrier()
    _remote_read_kernel[(N, )](out_ptr=out, device_dptr=device_dptr, MY_RANK=rank, N_RANKS=world_size)
    torch.cuda.synchronize()
    assert torch.allclose(out, expected), (f"rank {rank}: constant peer read returned {out[:4].tolist()}, "
                                           f"expected {expected[:4].tolist()}")

    # Runtime peer: the shard id comes from tle.shard_id, so it reaches the
    # FlagCX call as a register rather than a folded constant.
    mesh = tle.device_mesh(tle.MeshConfig(device=world_size))
    out = torch.zeros(N, dtype=torch.float32, device="cuda")
    dist.barrier()
    _remote_read_runtime_peer_kernel[(N, )](out_ptr=out, device_dptr=device_dptr, mesh=mesh)
    torch.cuda.synchronize()
    assert torch.allclose(out, expected), (f"rank {rank}: runtime peer read returned {out[:4].tolist()}, "
                                           f"expected {expected[:4].tolist()}")

    print(f"[Rank {rank}] read peer rank {peer_rank}, sample={out[:4].tolist()}", flush=True)

    tle.cleanup_communicator()


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 2,
    reason="needs two devices under torchrun",
)
def test_device_remote_reads_peer_memory():
    _run_remote_read_check()


if __name__ == "__main__":
    _run_remote_read_check()
