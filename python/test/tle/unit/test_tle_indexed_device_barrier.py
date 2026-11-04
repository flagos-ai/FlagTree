# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""CPU compilation and host resource contracts for independent device slots."""
from types import SimpleNamespace

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.experimental.tle.language import communication


@pytest.fixture(autouse=True)
def _nvidia_codegen_backend(monkeypatch):
    # These tests explicitly compile CUDA targets, including without a visible GPU.
    from triton import _flagtree_backend
    monkeypatch.setattr(_flagtree_backend, "FLAGTREE_BACKEND", "nvidia")


@triton.jit
def _indexed(mem: tl.int64, comm: tl.int64, COUNT: tl.constexpr):
    tle.distributed_barrier(device_dptr=(mem, comm), space="device", index=1 + tl.program_id(0), barrier_count=COUNT)


@triton.jit
def _invalid_index(mem: tl.int64, comm: tl.int64, KIND: tl.constexpr):
    if KIND == 0:
        index = tl.program_id(0).to(tl.int64)
    elif KIND == 1:
        index = tl.arange(0, 4)
    else:
        index = 17
    tle.distributed_barrier(device_dptr=(mem, comm), space="device", index=index, barrier_count=17)


def test_indexed_barrier_compiles_without_cuda_context():
    compiled = triton.compile(ASTSource(_indexed, signature={"mem": "i64", "comm": "i64"}, constexprs={"COUNT": 17}),
                              target=GPUTarget("cuda", 90, 32), options={"num_warps": 4})
    assert '"tle.device intra barrier"' in compiled.asm["ttir"]
    assert 'barrier_count = 17' in compiled.asm["ttir"]
    assert "flagcxIntraBarrierSyncS" in compiled.asm["llir"]


@triton.jit
def _static_single_channel(mem: tl.int64, comm: tl.int64):
    tle.distributed_barrier(device_dptr=(mem, comm), space="device")


def test_meshless_single_channel_retains_the_local_communicator_abi():
    compiled = triton.compile(ASTSource(_static_single_channel, signature={"mem": "i64", "comm": "i64"}),
                              target=GPUTarget("cuda", 90, 32), options={"num_warps": 4})
    assert '"tle.device intra barrier"' in compiled.asm["ttir"]
    assert 'barrier_count = 1' in compiled.asm["ttir"]
    assert 'flagcxIntraBarrierSyncS' in compiled.asm["llir"]


@pytest.mark.parametrize("kind,match", [(0, "scalar int32"), (1, "scalar int32")])
def test_reject_wrong_index_representation_or_bounds(kind, match):
    with pytest.raises(triton.CompilationError, match=match):
        triton.compile(ASTSource(_invalid_index, signature={"mem": "i64", "comm": "i64"}, constexprs={"KIND": kind}),
                       target=GPUTarget("cuda", 90, 32), options={"num_warps": 4})


def test_constant_tensor_index_is_checked_by_ir_verifier(capfd):
    with pytest.raises(RuntimeError, match="parsing"):
        triton.compile(ASTSource(_invalid_index, signature={"mem": "i64", "comm": "i64"}, constexprs={"KIND": 2}),
                       target=GPUTarget("cuda", 90, 32), options={"num_warps": 4})
    assert "constant barrier index is outside the declared capacity" in capfd.readouterr().err


@pytest.mark.parametrize("count", [None, 0, -1, True, 2**31])
def test_reject_missing_or_invalid_capacity(count):
    with pytest.raises(triton.CompilationError, match="positive int32 barrier_count"):
        triton.compile(ASTSource(_indexed, signature={"mem": "i64", "comm": "i64"}, constexprs={"COUNT": count}),
                       target=GPUTarget("cuda", 90, 32), options={"num_warps": 4})


@pytest.mark.parametrize("count", [0, -1, True, 1.5, 2**31])
def test_host_rejects_capacity_before_cuda_side_effects(count):
    with pytest.raises(ValueError, match="positive int32"):
        communication.create_dist_tensor(None, intra_barrier_count=count)


def _host_environment(monkeypatch):
    requests = []

    class FlagCX:
        supports_intra_barrier_count_query = True

        def flagcxDevCommGetIntraBarrierCount(self, dev_comm):
            return requests[-1]

        def flagcxDevCommCreate(self, comm, requirements):
            requests.append(requirements.intraBarrierCount)
            return object()

        def __getattr__(self, name):
            return lambda *args, **kwargs: SimpleNamespace(value=1234)

    for name, value in dict(comm=object(), rank=0, dev_comm=None, dev_mem=None, win=None, flagcx=FlagCX()).items():
        monkeypatch.setattr(communication, name, value, raising=False)
    monkeypatch.setattr(communication, "_distributed_barrier", lambda: None)
    monkeypatch.setattr(communication, "torch", SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda *args: None)))
    monkeypatch.setattr(communication, "dist", SimpleNamespace(is_initialized=lambda: False))
    import triton.runtime
    monkeypatch.setattr(triton.runtime, "DistributedRtContext", lambda *args: SimpleNamespace())
    buffer = SimpleNamespace(is_cuda=True, is_contiguous=lambda: True, data_ptr=lambda: 4096, numel=lambda: 4096,
                             element_size=lambda: 1, device="cuda:0")
    return buffer, requests


def test_host_allocates_requested_runtime_slots(monkeypatch):
    buffer, requests = _host_environment(monkeypatch)
    context = communication.create_dist_tensor(buffer, intra_barrier_count=17)
    assert requests == [17]
    assert context.intra_barrier_count == 17


def test_active_context_cannot_be_replaced_with_different_capacity(monkeypatch):
    buffer, requests = _host_environment(monkeypatch)
    context = communication.create_dist_tensor(buffer, intra_barrier_count=1)
    with pytest.raises(RuntimeError, match="active FlagCX"):
        communication.create_dist_tensor(buffer, intra_barrier_count=17)
    assert requests == [1]
    assert context.intra_barrier_count == 1
