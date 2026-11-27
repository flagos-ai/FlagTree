"""Minimal dual-GPU remote load smoke (no barrier)."""
import triton.experimental.tle.language as tle
import torch
import triton
import triton.language as tl
import torch.distributed as dist
import os

DEVICE = tle.device_type
DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=2))
N = 8


@triton.jit(do_not_specialize=["mem_ptr", "comm_ptr"])
def _remote_only(out_ptr, mem_ptr, comm_ptr, mesh: tl.constexpr):
    device_dptr = (mem_ptr, comm_ptr)
    pid = tl.program_id(0)
    local_rank = tle.shard_id(mesh, "device", device_dptr=device_dptr)
    peer = (local_rank + 1) % mesh.shape[0]
    remote_mem = tle.remote(device_dptr, space="device", dtype=tl.float32, shard_id=peer, offset=pid)
    tl.store(out_ptr + pid, tl.load(remote_mem))


def main():
    local = int(os.environ.get("LOCAL_RANK", 0))
    # Best-effort peer access; FlagCX IPC still works when this API is absent.
    for peer in range(torch.musa.device_count()):
        if peer == local:
            continue
        try:
            torch.musa.device(local).enable_peer_access(peer)
        except Exception as e:
            print(f"peer access {local}->{peer}: {e}")

    mem_pool = tle.get_mem_pool()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    with torch.get_device_module(DEVICE).use_mem_pool(mem_pool):
        x = (torch.arange(N, dtype=torch.float32, device=DEVICE) + rank * 1000).clone()
    device_dptr = tle.create_dist_tensor(x)
    mem_ptr, comm_ptr = tle.runtime_device_dptr(device_dptr)
    output = torch.zeros(N, dtype=torch.float32, device=DEVICE)
    dist.barrier()
    _remote_only[(N,)](out_ptr=output, mem_ptr=mem_ptr, comm_ptr=comm_ptr, mesh=DEVICE_MESH)
    torch.get_device_module(DEVICE).synchronize()
    peer_rank = (rank + 1) % world_size
    expected = torch.arange(N, dtype=torch.float32, device=DEVICE) + peer_rank * 1000
    print(f"[Rank {rank}] out={output[:4].tolist()} exp={expected[:4].tolist()} ok={torch.allclose(output, expected)}")
    tle.cleanup_communicator()


if __name__ == "__main__":
    main()
