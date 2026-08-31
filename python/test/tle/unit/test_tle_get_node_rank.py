import os

import pytest
import torch
import torch.distributed as dist
import triton
import triton.experimental.tle.language as tle

import triton.language as tl

DEVICE = tle.device_type

LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
if WORLD_SIZE % LOCAL_WORLD_SIZE != 0:
    raise ValueError("WORLD_SIZE must be divisible by LOCAL_WORLD_SIZE")

DEVICE_MESH = tle.device_mesh(tle.MeshConfig(node=WORLD_SIZE // LOCAL_WORLD_SIZE, device=LOCAL_WORLD_SIZE))


@triton.jit
def _tle_node_rank_kernel(out_ptr, device_dptr: tl.constexpr, mesh: tl.constexpr):
    pid = tl.program_id(0)
    node_rank = tle.shard_id(mesh, "node", device_dptr=device_dptr)
    tl.store(out_ptr + pid, node_rank)


@pytest.mark.require_tle("shard_id")
def test_tle_get_node_rank():
    grid = 2
    with torch.get_device_module().use_mem_pool(tle.get_mem_pool()):
        source = torch.empty((1, ), dtype=torch.float32, device=DEVICE)
    device_dptr = tle.create_dist_tensor(source)
    node_rank_out = torch.empty((grid, ), dtype=torch.int32, device=DEVICE)

    compiled = _tle_node_rank_kernel.warmup(
        out_ptr=node_rank_out,
        device_dptr=device_dptr,
        mesh=DEVICE_MESH,
        grid=(grid, ),
        num_ctas=1,
        num_warps=4,
    )
    assert "get_world_rank" in compiled.asm["ttgir"]
    assert "get_num_pes" in compiled.asm["ttgir"]
    assert "flagcxDevCommGetRank" in compiled.asm["ptx"]
    assert "flagcxDevCommGetIntraSize" in compiled.asm["ptx"]

    _tle_node_rank_kernel[(grid, )](
        out_ptr=node_rank_out,
        device_dptr=device_dptr,
        mesh=DEVICE_MESH,
    )
    torch.get_device_module().synchronize()

    rank = dist.get_rank()
    expected_node_rank = rank // LOCAL_WORLD_SIZE
    actual_node_ranks = node_rank_out.cpu().tolist()
    try:
        torch.testing.assert_close(
            node_rank_out,
            torch.full_like(node_rank_out, expected_node_rank),
        )
    except AssertionError:
        print(
            f"[Rank {rank}] FAILED: node ranks={actual_node_ranks}, "
            f"expected={expected_node_rank}",
            flush=True,
        )
        raise
    else:
        print(
            f"[Rank {rank}] PASSED: node ranks={actual_node_ranks}, "
            f"expected={expected_node_rank}",
            flush=True,
        )
    finally:
        tle.cleanup_communicator()


if __name__ == "__main__":
    test_tle_get_node_rank()
