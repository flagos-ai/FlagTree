# Copyright 2026- Xcoresigma Technology Co., Ltd

import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton.experimental.tle as tle

DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=2))
N = 8


@triton.jit()
def _barrier_d2d_kernel(out_ptr, device_dptr, mesh: tl.constexpr):
    pid = tl.program_id(0)
    local_rank = tle.shard_id(mesh, 'device', device_dptr=device_dptr)
    n_rank = mesh.shape[0]
    peer = (local_rank + 1) % n_rank

    remote_mem = tle.remote(
        device_dptr,
        space="device",
        dtype=tl.float32,
        shard_id=peer,
        # offset=pid
    )
    val = tl.load(remote_mem + pid)
    tl.store(out_ptr + pid, val)
    tle.distributed_barrier(mesh=mesh, device_dptr=device_dptr, space="device")


def _runtime_verify(output, device_dptr, grid, rank, world_size):
    dist.barrier()
    _barrier_d2d_kernel[grid](device_dptr=device_dptr, out_ptr=output, mesh=DEVICE_MESH)

    torch.npu.synchronize()

    import sys
    peer_rank = (rank + 1) % world_size
    expected = torch.arange(N, dtype=torch.float32, device='npu') + peer_rank * 1000
    if torch.allclose(output, expected):
        print(f"[Rank {rank}] [PASSED] read peer rank {peer_rank}")
    else:
        print(f"[Rank {rank}] [FAILED] read peer rank {peer_rank}")
        print(f"[Rank {rank}] expected[:4] = {expected[:4].cpu().tolist()}")
        print(f"[Rank {rank}] output[:4] = {output[:4].cpu().tolist()}")
        sys.exit(1)

    tle.cleanup_communicator(device_dptr)


class TestD2DBarrier:

    def test_tle_d2d_barrier(self):
        grid = (N, )

        tle.init_communicator()
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        x = (torch.arange(N, dtype=torch.float32, device="npu") + rank * 1000).clone()
        device_dptr = tle.create_dist_tensor(x)
        device_dptr.copy_(x)
        output = torch.zeros(N, dtype=torch.float32, device="npu")

        _runtime_verify(output, device_dptr, grid, rank, world_size)


# cmd: torchrun --nproc-per-node=2 python/python/tutorials/tle/dsa/10-tle-d2d-barrier.py
if __name__ == "__main__":
    TestD2DBarrier().test_tle_d2d_barrier()
