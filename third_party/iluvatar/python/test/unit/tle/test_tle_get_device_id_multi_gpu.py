"""Multi-device check that get_device_id returns the real FlagCX intra rank.

The compile-only coverage lives in test_tle_get_device_id.py. This file closes
the gap that test leaves open: the kernel has to run and its result has to be
read back, because a get_device_id kernel that faults or hangs still produces
the expected IR.

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

GRID = 2
N = 64
DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=2))


@triton.jit
def _store_local_rank_kernel(out_ptr, device_dptr: tl.constexpr, mesh: tl.constexpr):
    pid = tl.program_id(0)
    local_rank = tle.shard_id(mesh, 'device', device_dptr=device_dptr)
    tl.store(out_ptr + pid, local_rank)


def _run_local_rank_check():
    import torch.distributed as dist

    # Bringing up the memory pool initializes the FlagCX communicator, so the
    # rank is only available afterwards.
    mem_pool = tle.get_mem_pool()
    rank = dist.get_rank()
    with torch.cuda.use_mem_pool(mem_pool):
        x = torch.randn((N, N), dtype=torch.float32, device="cuda")
    device_dptr = tle.create_dist_tensor(x)

    out = torch.full((GRID, ), -1, dtype=torch.int32, device="cuda")
    _store_local_rank_kernel[(GRID, )](out_ptr=out, device_dptr=device_dptr, mesh=DEVICE_MESH)
    torch.cuda.synchronize()

    got = out.cpu().tolist()
    assert got == [rank] * GRID, f"rank {rank}: expected {[rank] * GRID}, got {got}"

    tle.cleanup_communicator()


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 2,
    reason="needs two devices under torchrun",
)
def test_get_device_id_matches_distributed_rank():
    _run_local_rank_check()


if __name__ == "__main__":
    _run_local_rank_check()
