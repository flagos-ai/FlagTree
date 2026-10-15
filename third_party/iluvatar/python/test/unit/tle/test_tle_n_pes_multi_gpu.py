"""Multi-device check that n_pes returns the real FlagCX intra size.

The compile-only coverage lives in test_tle_n_pes.py. This file closes the gap
that test leaves open: the kernel has to run and its result has to be read back,
because an n_pes kernel that faults or returns a stale value still produces the
expected IR.

Launch it with torchrun on two devices (see third_party/iluvatar/test_triton.sh);
under a plain pytest run it skips, so it stays part of the single-device suite
without needing a distributed setup.
"""
import os

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.experimental.tle.language.distributed import _get_local_rank, n_pes

GRID = 2
N = 64


# n_pes reads dev_mem_ptr.handle directly, so unlike shard_id it takes the raw
# communicator pointer as a runtime i64 rather than a DistributedRtContext.
@triton.jit
def _store_n_pes_kernel(out_ptr, comm):
    pid = tl.program_id(0)
    tl.store(out_ptr + pid, n_pes(comm))


@triton.jit
def _store_peer_rank_kernel(out_ptr, comm):
    pid = tl.program_id(0)
    rank = _get_local_rank(comm)
    peer = (rank + 1) % n_pes(comm)
    tl.store(out_ptr + pid, peer)


def _run_n_pes_check():
    import torch.distributed as dist

    # Bringing up the memory pool initializes the FlagCX communicator, so the
    # intra size is only available afterwards.
    mem_pool = tle.get_mem_pool()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    with torch.cuda.use_mem_pool(mem_pool):
        x = torch.randn((N, N), dtype=torch.float32, device="cuda")
    # Index 1 of the runtime context is the communicator pointer; this is the
    # same slot _parse_src_arg feeds to shard_id/_get_local_rank.
    comm = int(tle.create_dist_tensor(x)[1])

    out = torch.full((GRID, ), -1, dtype=torch.int32, device="cuda")
    _store_n_pes_kernel[(GRID, )](out_ptr=out, comm=comm)
    torch.cuda.synchronize()

    got = out.cpu().tolist()
    assert got == [world_size] * GRID, f"rank {rank}: expected {[world_size] * GRID}, got {got}"

    peers = torch.full((GRID, ), -1, dtype=torch.int32, device="cuda")
    _store_peer_rank_kernel[(GRID, )](out_ptr=peers, comm=comm)
    torch.cuda.synchronize()

    expected_peer = (rank + 1) % world_size
    got_peers = peers.cpu().tolist()
    assert got_peers == [expected_peer] * GRID, f"rank {rank}: expected {[expected_peer] * GRID}, got {got_peers}"

    tle.cleanup_communicator()


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 2,
    reason="needs two devices under torchrun",
)
def test_n_pes_matches_distributed_world_size():
    _run_n_pes_check()


if __name__ == "__main__":
    _run_n_pes_check()
