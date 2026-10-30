"""Check the XPU TLE peer count and derived peer rank on two devices."""

import os

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from triton.backends.xpu.tle_supported import TLE_SUPPORTED_PRIMITIVES

pytestmark = [
    pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="needs two devices under torchrun"),
    pytest.mark.skipif("shard_id" not in TLE_SUPPORTED_PRIMITIVES or not hasattr(tle, "n_pes"),
                       reason="XPU TLE device rank or n_pes is not implemented"),
    pytest.mark.skipif(
        not all(hasattr(tle, name) for name in ("get_mem_pool", "create_dist_tensor", "cleanup_communicator")),
        reason="XPU TLE FlagCX communicator API is not available"),
]

DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=2))


@triton.jit
def _store_num_pes(out_ptr, comm):
    tl.store(out_ptr + tl.program_id(0), tle.n_pes(comm))


@triton.jit
def _store_peer_rank(out_ptr, comm, mesh: tl.constexpr):
    rank = tle.shard_id(mesh, "device", device_dptr=comm)
    tl.store(out_ptr + tl.program_id(0), (rank + 1) % tle.n_pes(comm))


def test_n_pes_matches_distributed_world_size():
    import torch.distributed as dist

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    pool = tle.get_mem_pool()
    with torch.cuda.use_mem_pool(pool):
        tensor = torch.randn((64, 64), dtype=torch.float32, device="cuda")
    context = tle.create_dist_tensor(tensor)
    comm = int(context[1])
    try:
        count = torch.full((2, ), -1, dtype=torch.int32, device="cuda")
        peer = torch.full_like(count, -1)
        _store_num_pes[(2, )](count, comm)
        _store_peer_rank[(2, )](peer, comm, DEVICE_MESH)
        torch.cuda.synchronize()
        assert count.cpu().tolist() == [world_size, world_size]
        assert peer.cpu().tolist() == [(rank + 1) % world_size] * 2
    finally:
        tle.cleanup_communicator()
