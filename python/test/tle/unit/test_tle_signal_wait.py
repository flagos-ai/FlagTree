import os

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

DEVICE = tle.device_type

LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", "2"))
DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=LOCAL_WORLD_SIZE))


@triton.jit()
def _signal_wait_kernel(
    result_ptr,
    device_dptr: tl.constexpr,
    mesh: tl.constexpr,
    peer: tl.constexpr,
    slot_id: tl.constexpr,
    target: tl.constexpr | None,
    signal_space: tl.constexpr,
    signal_op: tl.constexpr,
    wait_kind: tl.constexpr,
):
    local_rank = tle.shard_id(mesh, "device", device_dptr=device_dptr)

    if signal_op is not None:
        tle.signal(
            device_dptr,
            peer,
            slot_id=slot_id,
            op=signal_op,
            space=signal_space,
            group_kind="block",
            context_idx=0,
        )
    tle.signal_wait(
        device_dptr,
        slot_id=slot_id,
        wait_kind=wait_kind,
        target=target,
        group_kind="block",
        context_idx=0,
    )
    tl.store(result_ptr, local_rank + 1)


def _ir_verify(
    result,
    device_dptr,
    peer,
    slot_id,
    target,
    signal_space,
    signal_op,
    wait_kind,
):
    compiled = _signal_wait_kernel.warmup(
        result_ptr=result,
        device_dptr=device_dptr,
        mesh=DEVICE_MESH,
        peer=peer,
        slot_id=slot_id,
        target=target,
        signal_space=signal_space,
        signal_op=signal_op,
        wait_kind=wait_kind,
        grid=(1, ),
        num_ctas=1,
        num_warps=4,
    )
    if signal_op is not None:
        assert "tle.signal" in compiled.asm["ttgir"]
        expected_signal_func = {
            "inc": "flagcxDevSignalInc",
            "add": "flagcxDevSignalAdd",
        }[signal_op]
        assert expected_signal_func in compiled.asm["ptx"]
    assert "tle.signal_wait" in compiled.asm["ttgir"]
    expected_wait_func = {
        "signal": "flagcxDevWaitSignal",
        "shadow": "flagcxDevWaitSignalMeetShadow",
        "counter": "flagcxDevWaitCounter",
    }[wait_kind]
    assert expected_wait_func in compiled.asm["ptx"]


def _runtime_verify(
    result,
    device_dptr,
    peer,
    rank,
    local_rank,
    slot_id,
    target,
    signal_space,
    signal_op,
    wait_kind,
):
    dist.barrier()
    _signal_wait_kernel[(1, )](
        result_ptr=result,
        device_dptr=device_dptr,
        mesh=DEVICE_MESH,
        peer=peer,
        slot_id=slot_id,
        target=target,
        signal_space=signal_space,
        signal_op=signal_op,
        wait_kind=wait_kind,
        num_ctas=1,
        num_warps=4,
    )
    torch.get_device_module().synchronize()

    actual = result.item()
    expected = local_rank + 1
    print(
        f"[Rank {rank}, local_rank={local_rank}] space={signal_space}, "
        f"peer={peer}, op={signal_op}, wait_kind={wait_kind}, "
        f"slot_id={slot_id}, result={actual}, "
        f"expected={expected}",
        flush=True,
    )
    passed = torch.tensor(
        int(actual == expected),
        dtype=torch.int32,
        device=DEVICE,
    )
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    assert passed.item() == 1, (f"[Rank {rank}, local_rank={local_rank}] "
                                f"{signal_space} {signal_op}/{wait_kind} failed: "
                                f"expected {expected}, got {actual}")
    print(
        f"[Rank {rank}] [PASSED] space={signal_space} {signal_op}/{wait_kind} kernel executed successfully",
        flush=True,
    )


@triton.jit()
def _signal_wait_verifier_kernel(
    device_dptr: tl.constexpr,
    mesh: tl.constexpr,
    peer: tl.constexpr,
    signal_op: tl.constexpr,
    value: tl.constexpr,
    wait_kind: tl.constexpr,
    target: tl.constexpr,
):
    tle.signal(device_dptr, peer, slot_id=0, op=signal_op, value=value, space="world", group_kind="block",
               context_idx=0)
    tle.signal_wait(device_dptr, slot_id=0, wait_kind=wait_kind, target=target, group_kind="block", context_idx=0)


def _verifier_verify(device_dptr, peer):
    phases = (
        ("inc", 1, "signal", 0),
        ("add", None, "signal", 0),
        ("inc", None, "signal", None),
        ("inc", None, "counter", None),
        ("inc", None, "shadow", 0),
    )
    for signal_op, value, wait_kind, target in phases:
        try:
            _signal_wait_verifier_kernel[(1, )](
                device_dptr,
                DEVICE_MESH,
                peer,
                signal_op,
                value,
                wait_kind,
                target,
                num_ctas=1,
                num_warps=4,
            )
            assert False, "illegal arg to op: should fail to compile"
        except ValueError:
            pass
        except triton.compiler.errors.CompilationError:
            pass


class TestSignalWait:

    @pytest.mark.skipif(
        "NODE_RANK" not in os.environ,
        reason="requires torchrun and a configured FlagCX runtime",
    )
    def test_tle_signal_wait(self):
        mem_pool = tle.get_mem_pool()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % LOCAL_WORLD_SIZE))
        node_rank = int(os.environ.get(
            "GROUP_RANK",
            os.environ.get("NODE_RANK", str(rank // LOCAL_WORLD_SIZE)),
        ))
        world_size = dist.get_world_size()
        nnodes = (world_size + LOCAL_WORLD_SIZE - 1) // LOCAL_WORLD_SIZE
        signal_target = 1 if nnodes > 1 else 0
        assert world_size == nnodes * LOCAL_WORLD_SIZE, (
            "signal_wait test requires the same LOCAL_WORLD_SIZE on every node")

        inter_node_peer = (node_rank + 1) % nnodes
        world_peer = inter_node_peer * LOCAL_WORLD_SIZE + local_rank

        print(
            f"[Rank {rank}] node_rank={node_rank}, local_rank={local_rank}, "
            f"inter_node_peer={inter_node_peer}, world_peer={world_peer}, nnodes={nnodes}, "
            f"local_world_size={LOCAL_WORLD_SIZE}",
            flush=True,
        )
        with torch.get_device_module().use_mem_pool(mem_pool):
            backing = torch.tensor([rank], dtype=torch.int32, device=DEVICE)

        device_dptr = tle.create_dist_tensor(backing)
        inter_node_result = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        world_result = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        counter_result = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        shadow_result = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        try:
            phases = (
                (
                    inter_node_result,
                    inter_node_peer,
                    0,
                    signal_target,
                    "inter_node" if nnodes > 1 else "world",
                    "inc",
                    "signal",
                ),
                (world_result, world_peer, 1, signal_target, "world", "inc", "signal"),
                (counter_result, world_peer, 2, 0, "world", None, "counter"),
                (shadow_result, world_peer, 3, None, "world", None, "shadow"),
            )
            for result, peer, slot_id, target, signal_space, signal_op, wait_kind in phases:
                _ir_verify(
                    result,
                    device_dptr,
                    peer,
                    slot_id,
                    target,
                    signal_space,
                    signal_op,
                    wait_kind,
                )
                _runtime_verify(
                    result,
                    device_dptr,
                    peer,
                    rank,
                    local_rank,
                    slot_id,
                    target,
                    signal_space,
                    signal_op,
                    wait_kind,
                )
            _verifier_verify(device_dptr, world_peer)
        finally:
            tle.cleanup_communicator()


if __name__ == "__main__":
    TestSignalWait().test_tle_signal_wait()
