"""Exercise FlagCX remote stores and the device barrier with two local torchrun ranks.

Set up the XPU FlagCX runtime and prebuild the allocator in the ranks' shared
temporary directory before launching the test.
"""

import os

import pytest
import torch
import torch.distributed as dist
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=2))
N = 8


@triton.jit
def _store_and_barrier(device_dptr: tl.constexpr, mesh: tl.constexpr):
    tle.init(device_dptr)
    offset = tl.program_id(0)
    rank = tle.shard_id(mesh, "device", device_dptr=device_dptr)
    peer = (rank + 1) % mesh.shape[0]
    remote_ptr = tle.remote(device_dptr, space="device", dtype=tl.float32, shard_id=peer, offset=offset)
    if tl.cast(remote_ptr, tl.uint64) != 0:
        tl.store(remote_ptr, tl.cast(rank * 1000 + offset, tl.float32))
    tle.distributed_barrier(mesh, device_dptr=device_dptr, space="device")


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2" or os.environ.get("LOCAL_WORLD_SIZE") != "2",
    reason="requires two local torchrun ranks with the XPU FlagCX runtime",
)
def test_tle_d2d_barrier():
    mem_pool = tle.get_mem_pool()
    rank = dist.get_rank()
    with torch.cuda.use_mem_pool(mem_pool):
        receive = torch.full((N,), -1.0, dtype=torch.float32).to("cuda")
    device_dptr = tle.create_dist_tensor(receive)

    try:
        print(f"[Rank {rank}] launching remote stores and distributed_barrier", flush=True)
        compiled = _store_and_barrier[(N, )](device_dptr, DEVICE_MESH, num_ctas=1, num_warps=4)
        torch.cuda.synchronize()

        assert compiled.metadata.xpu_uses_flagcx
        ttxir = compiled.asm.get("ttxir")
        if ttxir:
            assert "triton_xpu.remote_pointer" in ttxir
            assert "triton_xpu.distributed_barrier" in ttxir
        peer = (rank + 1) % dist.get_world_size()
        actual = receive.cpu()
        expected = torch.arange(N, dtype=torch.float32) + peer * 1000
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print(f"[Rank {rank}] PASSED: received {actual.tolist()} from rank {peer}", flush=True)
    finally:
        tle.cleanup_communicator()


if __name__ == "__main__":
    test_tle_d2d_barrier()
