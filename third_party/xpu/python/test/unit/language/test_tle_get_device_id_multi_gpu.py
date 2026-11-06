"""Check that the XPU TLE device rank matches the distributed process rank."""

import os

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from triton.backends.xpu.tle_supported import TLE_SUPPORTED_PRIMITIVES

pytestmark = [
    pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="needs two devices under torchrun"),
    pytest.mark.skipif("shard_id" not in TLE_SUPPORTED_PRIMITIVES,
                       reason="XPU TLE device-rank lowering has not been enabled"),
    pytest.mark.skipif(
        not all(hasattr(tle, name) for name in ("get_mem_pool", "create_dist_tensor", "cleanup_communicator")),
        reason="XPU TLE FlagCX communicator API is not available"),
]

DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=2))


@triton.jit
def _store_local_rank(out_ptr, device_dptr: tl.constexpr, mesh: tl.constexpr):
    tl.store(out_ptr + tl.program_id(0), tle.shard_id(mesh, "device", device_dptr=device_dptr))


def test_get_device_id_matches_distributed_rank():
    import torch.distributed as dist

    rank = dist.get_rank()
    pool = tle.get_mem_pool()
    with torch.cuda.use_mem_pool(pool):
        tensor = torch.randn((64, 64), dtype=torch.float32, device="cuda")
    context = tle.create_dist_tensor(tensor)
    try:
        out = torch.full((2, ), -1, dtype=torch.int32, device="cuda")
        _store_local_rank[(2, )](out, context, DEVICE_MESH)
        torch.cuda.synchronize()
        assert out.cpu().tolist() == [rank, rank]
    finally:
        tle.cleanup_communicator()
