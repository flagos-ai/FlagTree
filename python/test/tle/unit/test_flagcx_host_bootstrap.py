# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Host rendezvous must not allocate an unused CUDA NCCL communicator."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def communication(monkeypatch):
    from triton.experimental.tle.language import communication as comm

    if not comm.enabled:
        pytest.skip("FlagCX is unavailable")
    monkeypatch.setattr(comm, "_init_communicator_", False)
    monkeypatch.setattr(comm, "_owns_process_group", False)
    monkeypatch.setattr(comm, "_sync_group", object())
    monkeypatch.setattr(comm, "comm", None)
    monkeypatch.setattr(comm, "rank", 0, raising=False)
    monkeypatch.setattr(comm, "dev_mem", None)
    monkeypatch.setattr(comm, "dev_comm", None)
    monkeypatch.setattr(comm, "win", None)
    monkeypatch.setattr(comm.torch.cuda, "set_device", lambda *_: None)
    return comm


@pytest.mark.parametrize("rank", (0, 1))
def test_unique_id_uses_cpu_tensor_and_the_host_group(communication, monkeypatch, rank):
    comm = communication
    broadcasts = []

    def broadcast(tensor, *, src, group=None):
        assert tensor.device.type == "cpu", "FlagCX ID bootstrap must stay on the host"
        assert group is comm._sync_group
        tensor.copy_(comm.torch.arange(256, dtype=comm.torch.uint8))
        broadcasts.append(bytes(tensor.numpy()))

    def forbidden_cuda(*_args, **_kwargs):
        pytest.fail("FlagCX ID bootstrap must not initialize CUDA NCCL")

    monkeypatch.setattr(comm.torch.Tensor, "cuda", forbidden_cuda)
    monkeypatch.setattr(comm, "dist", SimpleNamespace(
        is_initialized=lambda: True, get_rank=lambda: rank, get_world_size=lambda: 2,
        broadcast=broadcast))
    identifier = bytes(range(256))
    token = object()
    monkeypatch.setattr(comm, "flagcx", SimpleNamespace(
        flagcxGetUniqueId=lambda: SimpleNamespace(internal=identifier),
        unique_id_from_bytes=lambda data: SimpleNamespace(internal=data),
        flagcxCommInitRank=lambda *_: token))
    comm.init_communicator()
    assert broadcasts == [identifier]
    assert comm.comm is token
    assert comm._init_communicator_
    assert not comm._owns_process_group


def test_window_publication_synchronizes_cuda_then_uses_host_barriers(communication, monkeypatch):
    import triton.runtime

    comm = communication
    events = []

    def barrier(*, group=None, **kwargs):
        assert group is comm._sync_group, "Window publication must use the host group"
        assert not kwargs
        events.append("host_barrier")

    def register(*_args, **_kwargs):
        events.append("register")
        return object()

    monkeypatch.setattr(comm.torch.cuda, "synchronize", lambda *_: events.append("cuda_sync"))
    monkeypatch.setattr(comm, "dist", SimpleNamespace(is_initialized=lambda: True, barrier=barrier))
    monkeypatch.setattr(comm, "flagcx", SimpleNamespace(
        flagcxCommWindowRegister=register,
        flagcxDevCommCreate=lambda *_: object(), flagcxDevMemCreate=lambda *_: object(),
        flagcxDevCommGetDevicePtr=lambda *_: SimpleNamespace(value=11),
        flagcxDevMemGetDevicePtr=lambda *_: SimpleNamespace(value=22)))
    monkeypatch.setattr(triton.runtime, "DistributedRtContext", lambda *_: "context")
    window = SimpleNamespace(is_cuda=True, is_contiguous=lambda: True, data_ptr=lambda: 4096,
                             numel=lambda: 4096, element_size=lambda: 1, device="cuda:0")
    assert comm.create_dist_tensor(window) == "context"
    assert events == ["cuda_sync", "host_barrier", "register", "host_barrier"]
