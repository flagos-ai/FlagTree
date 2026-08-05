import os

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", "2"))
DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=LOCAL_WORLD_SIZE))


@triton.jit()
def _signal_kernel(
    result_ptr,
    device_dptr: tl.constexpr,
    mesh: tl.constexpr,
    peer: tl.constexpr,
    world_peer: tl.constexpr,
):
    local_rank = tle.shard_id(mesh, "device", device_dptr=device_dptr)

    tle.signal(
        device_dptr,
        peer,
        signal_id=0,
        op="inc",
        space="inter_node",
        group_kind="block",
        context_idx=0,
    )
    tle.signal(
        device_dptr,
        peer,
        signal_id=1,
        value=local_rank + 2,
        op="add",
        space="inter_node",
        group_kind="block",
        context_idx=1,
    )
    tle.signal(
        device_dptr,
        world_peer,
        signal_id=0,
        op="inc",
        space="world",
        group_kind="block",
        context_idx=0,
    )
    tl.store(result_ptr, local_rank + 1)


def _ir_verify(result, device_dptr, peer, world_peer):
    compiled = _signal_kernel.warmup(
        result_ptr=result,
        device_dptr=device_dptr,
        mesh=DEVICE_MESH,
        peer=peer,
        world_peer=world_peer,
        grid=(1, ),
        num_ctas=1,
        num_warps=4,
    )
    assert "tle.signal" in compiled.asm["ttgir"]
    assert "flagcxDevNetGetFromCommS" in compiled.asm["ptx"]
    assert "flagcxDevNetSignalSigIncS" in compiled.asm["ptx"]
    assert "flagcxDevNetSignalSigAddS" in compiled.asm["ptx"]


def _runtime_verify(result, device_dptr, peer, world_peer, rank, local_rank):
    dist.barrier()
    _signal_kernel[(1, )](
        result_ptr=result,
        device_dptr=device_dptr,
        mesh=DEVICE_MESH,
        peer=peer,
        world_peer=world_peer,
        num_ctas=1,
        num_warps=4,
    )
    torch.cuda.synchronize()

    actual = result.item()
    expected = local_rank + 1
    print(
        f"[Rank {rank}, local_rank={local_rank}] result={actual}, expected={expected}",
        flush=True,
    )
    passed = torch.tensor(
        int(actual == expected),
        dtype=torch.int32,
        device="cuda",
    )
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    assert passed.item() == 1, (
        f"[Rank {rank}, local_rank={local_rank}] signal failed: "
        f"expected {expected}, got {actual}"
    )
    print(f"[Rank {rank}] [PASSED] signal kernel executed successfully", flush=True)


class TestSignal:

    @pytest.mark.skipif(
        not tle.communication.enabled or "RANK" not in os.environ,
        reason="requires torchrun and a configured FlagCX runtime",
    )
    def test_tle_signal(self):
        mem_pool = tle.get_mem_pool()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % LOCAL_WORLD_SIZE))
        node_rank = int(
            os.environ.get(
                "GROUP_RANK",
                os.environ.get("NODE_RANK", str(rank // LOCAL_WORLD_SIZE)),
            )
        )
        nnodes = (dist.get_world_size() + LOCAL_WORLD_SIZE - 1) // LOCAL_WORLD_SIZE
        peer = (node_rank + 1) % nnodes
        world_peer = (rank + 1) % dist.get_world_size()

        print(
            f"[Rank {rank}] node_rank={node_rank}, local_rank={local_rank}, "
            f"peer_node={peer}, nnodes={nnodes}, "
            f"world_peer={world_peer}, "
            f"local_world_size={LOCAL_WORLD_SIZE}",
            flush=True,
        )
        with torch.cuda.use_mem_pool(mem_pool):
            backing = torch.tensor([rank], dtype=torch.int32, device="cuda")

        device_dptr = tle.create_dist_tensor(backing)
        result = torch.zeros(1, dtype=torch.int32, device="cuda")
        try:
            _ir_verify(result, device_dptr, peer, world_peer)
            _runtime_verify(result, device_dptr, peer, world_peer, rank, local_rank)
        finally:
            tle.cleanup_communicator()


if __name__ == "__main__":
    TestSignal().test_tle_signal()
