"""Read a registered peer buffer using constant and runtime XPU shard IDs."""

import os

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from triton.backends.xpu.tle_supported import TLE_SUPPORTED_PRIMITIVES

_HAS_COMMUNICATOR = all(hasattr(tle, name) for name in ("get_mem_pool", "create_dist_tensor", "cleanup_communicator"))
pytestmark = [
    pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="needs two devices under torchrun"),
    pytest.mark.skipif("remote" not in TLE_SUPPORTED_PRIMITIVES or not _HAS_COMMUNICATOR,
                       reason="XPU TLE remote pointer or FlagCX communicator API is not available"),
]

N = 64
RANK_STRIDE = 1000
DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=2))


@triton.jit
def _remote_read_constant_peer(out_ptr, device_dptr: tl.constexpr, PEER: tl.constexpr):
    pid = tl.program_id(0)
    remote_ptr = tle.remote(device_dptr, space="device", dtype=tl.float32, shard_id=PEER, offset=pid)
    tl.store(out_ptr + pid, tl.load(remote_ptr))


@triton.jit
def _remote_read_runtime_peer(out_ptr, device_dptr: tl.constexpr, mesh: tl.constexpr):
    pid = tl.program_id(0)
    rank = tle.shard_id(mesh, "device", device_dptr=device_dptr)
    peer = (rank + 1) % mesh.shape[0]
    remote_ptr = tle.remote(device_dptr, space="device", dtype=tl.float32, shard_id=peer, offset=pid)
    tl.store(out_ptr + pid, tl.load(remote_ptr))


def _registered_buffer():
    import torch.distributed as dist

    pool = tle.get_mem_pool()
    with torch.cuda.use_mem_pool(pool):
        buffer = (torch.arange(N, dtype=torch.float32, device="cuda") + dist.get_rank() * RANK_STRIDE).clone()
    return tle.create_dist_tensor(buffer)


def _check_read(launch, expected):
    import torch.distributed as dist

    out = torch.full((N, ), -1, dtype=torch.float32, device="cuda")
    dist.barrier()
    launch(out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_remote_reads_constant_peer():
    import torch.distributed as dist

    context = _registered_buffer()
    peer = (dist.get_rank() + 1) % 2
    expected = torch.arange(N, dtype=torch.float32, device="cuda") + peer * RANK_STRIDE
    try:
        _check_read(lambda out: _remote_read_constant_peer[(N, )](out, context, peer), expected)
    finally:
        tle.cleanup_communicator()


@pytest.mark.skipif("shard_id" not in TLE_SUPPORTED_PRIMITIVES,
                    reason="XPU TLE device-rank lowering has not been enabled")
def test_remote_reads_runtime_peer():
    import torch.distributed as dist

    context = _registered_buffer()
    peer = (dist.get_rank() + 1) % 2
    expected = torch.arange(N, dtype=torch.float32, device="cuda") + peer * RANK_STRIDE
    try:
        _check_read(lambda out: _remote_read_runtime_peer[(N, )](out, context, DEVICE_MESH), expected)
    finally:
        tle.cleanup_communicator()
