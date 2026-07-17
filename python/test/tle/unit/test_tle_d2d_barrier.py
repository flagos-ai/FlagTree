import triton.experimental.tle.language as tle
import torch
import triton
import triton.language as tl
import torch.distributed as dist

DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=2))
N = 8


@triton.jit()
def _barrier_d2d_kernel(out_ptr, device_dptr: tl.constexpr, mesh: tl.constexpr):
    pid = tl.program_id(0)
    local_rank = tle.shard_id(mesh, 'device', device_dptr=device_dptr)
    n_rank = mesh.shape[0]
    peer = (local_rank + 1) % n_rank  # noqa: F841

    remote_mem = tle.remote(
        device_dptr,
        space="device",
        dtype=tl.float32,
        shard_id=peer,
        offset=pid,
    )
    val = tl.load(remote_mem)
    tl.store(out_ptr + pid, val)
    tle.distributed_barrier(device_dptr=device_dptr, space="device")


def _ir_verify(output, device_dptr, grid):
    compiled = _barrier_d2d_kernel.warmup(
        device_dptr=device_dptr,
        out_ptr=output,
        mesh=DEVICE_MESH,
        grid=(grid, ),
        num_ctas=1,
        num_warps=4,
    )
    assert "distributed_barrier" in compiled.asm["ttgir"]
    assert "remote_pointer" in compiled.asm["ttgir"]
    assert "flagcxIntraBarrier" in compiled.asm['ptx']
    assert "flagcxGetIntraPointerC" in compiled.asm['ptx']
    assert "flagcxDevCommGetIntraRank" in compiled.asm['ptx']


def _runtime_verify(output, device_dptr, grid, rank, world_size):
    dist.barrier()
    _barrier_d2d_kernel[grid](device_dptr=device_dptr, out_ptr=output, mesh=DEVICE_MESH)

    torch.cuda.synchronize()

    import sys
    peer_rank = (rank + 1) % world_size
    expected = torch.arange(N, dtype=torch.float32, device="cuda") + peer_rank * 1000
    if torch.allclose(output, expected):
        print(f"[Rank {rank}] [PASSED] read peer rank {peer_rank}")
        print(f"[Rank {rank}] sample output[:4] = {output[:4].tolist()}")
    else:
        print(f"[Rank {rank}] [FAILED] read peer rank {peer_rank}")
        print(f"[Rank {rank}] expected[:4] = {expected[:4].tolist()}")
        print(f"[Rank {rank}] output[:4] = {output[:4].tolist()}")
        sys.exit(1)

    tle.cleanup_communicator()


class TestD2DBarrier:

    def test_tle_d2d_barrier(self):
        grid = (N, )

        mem_pool = tle.get_mem_pool()
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        with torch.cuda.use_mem_pool(mem_pool):
            x = (torch.arange(N, dtype=torch.float32, device="cuda") + rank * 1000).clone()

        device_dptr = tle.create_dist_tensor(x)
        output = torch.zeros(N, dtype=torch.float32, device="cuda")

        _ir_verify(output, device_dptr, grid)

        _runtime_verify(output, device_dptr, grid, rank, world_size)


if __name__ == "__main__":
    TestD2DBarrier().test_tle_d2d_barrier()
