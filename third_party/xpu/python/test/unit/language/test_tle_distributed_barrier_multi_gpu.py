"""Run one FlagCX intra-device barrier per rank on two XPU devices."""

import os

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from triton.backends.xpu.tle_supported import TLE_SUPPORTED_PRIMITIVES

pytestmark = [
    pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="needs two devices under torchrun"),
    pytest.mark.skipif("distributed_barrier" not in TLE_SUPPORTED_PRIMITIVES,
                       reason="XPU FlagCX device bitcode is not available"),
    pytest.mark.skipif(
        not all(hasattr(tle, name) for name in ("get_mem_pool", "create_dist_tensor", "cleanup_communicator")),
        reason="XPU FlagCX host communicator API is not available"),
]

DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=2))


@triton.jit
def _barrier_kernel(out, context: tl.constexpr, mesh: tl.constexpr):
    tle.distributed_barrier(mesh, context, space="device", group_kind="block")
    tl.store(out, 1)


def test_intra_device_barrier_completes():
    import torch.distributed as dist

    pool = tle.get_mem_pool()
    with torch.cuda.use_mem_pool(pool):
        registered = torch.zeros((64, 64), device="cuda")
    context = tle.create_dist_tensor(registered)
    try:
        out = torch.zeros((1, ), dtype=torch.int32, device="cuda")
        dist.barrier()
        _barrier_kernel[(1, )](out, context, DEVICE_MESH)
        torch.cuda.synchronize()
        assert out.item() == 1
    finally:
        tle.cleanup_communicator()
