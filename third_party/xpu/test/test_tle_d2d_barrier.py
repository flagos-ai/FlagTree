"""Exercise FlagCX remote stores and the device barrier with two local torchrun ranks.

Run test_tle_d2d_barrier.sh to configure the XPU runtime and prepare the
shared allocator before launching the ranks.
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


@triton.jit()
def _barrier_d2d_kernel(device_dptr, mesh: tl.constexpr):
    pid = tl.program_id(0)
    local_rank = tle.shard_id(mesh, "device", device_dptr=device_dptr)
    n_rank = mesh.shape[0]
    peer = (local_rank + 1) % n_rank

    remote_mem = tle.remote(
        device_dptr,
        space="device",
        dtype=tl.float32,
        shard_id=peer,
        offset=pid,
    )
    if tl.cast(remote_mem, tl.uint64) != 0:
        val = tl.cast(local_rank * 1000 + pid, tl.float32)
        tl.store(remote_mem, val)
    tle.distributed_barrier(mesh, device_dptr=device_dptr, space="device")


def _ir_verify(compiled):
    assert compiled.metadata.xpu_uses_flagcx
    ttxir = compiled.asm.get("ttxir")
    if ttxir:
        assert ttxir.count("triton_xpu.device_init") == 1
        assert ttxir.index("triton_xpu.device_init") < ttxir.index("scf.for")
        assert "triton_xpu.remote_pointer" in ttxir
        assert "triton_xpu.distributed_barrier" in ttxir
    llir = compiled.asm.get("llir")
    if llir:
        assert "@flagcxGetIntraPointerWithAccessC(" in llir


def _runtime_verify(output, device_dptr, grid, rank, world_size):
    print(f"[Rank {rank}] launching remote stores and distributed_barrier", flush=True)
    compiled = _barrier_d2d_kernel[grid](device_dptr=device_dptr, mesh=DEVICE_MESH, num_ctas=1, num_warps=4)
    torch.cuda.synchronize()
    _ir_verify(compiled)

    peer_rank = (rank - 1) % world_size
    actual = output.cpu()
    expected = torch.arange(N, dtype=torch.float32) + peer_rank * 1000
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(f"[Rank {rank}] PASSED: received {actual.tolist()} from rank {peer_rank}", flush=True)


class TestD2DBarrier:

    @pytest.mark.skipif(
        os.environ.get("WORLD_SIZE") != "2" or os.environ.get("LOCAL_WORLD_SIZE") != "2",
        reason="requires two local torchrun ranks with the XPU FlagCX runtime",
    )
    def test_tle_d2d_barrier(self):
        grid = (N, )

        mem_pool = tle.get_mem_pool()
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        with torch.cuda.use_mem_pool(mem_pool):
            output = torch.full((N, ), -1.0, dtype=torch.float32).to("cuda")
        device_dptr = tle.create_dist_tensor(output)

        try:
            _runtime_verify(output, device_dptr, grid, rank, world_size)
        finally:
            tle.cleanup_communicator()


if __name__ == "__main__":
    TestD2DBarrier().test_tle_d2d_barrier()
