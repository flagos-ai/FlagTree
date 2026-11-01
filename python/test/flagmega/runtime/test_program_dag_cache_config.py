# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Joint residency must retain one shared/L1 preference across DAG entries."""

import ctypes
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from triton.flagmega.errors import RuntimeContractError
from triton.flagmega.runtime.prepared import ResourceContract
from triton.flagmega.runtime.program_dag import ProgramDAG


class CarveoutSetter:
    def __init__(self, status=0):
        self.calls = []
        self.status = status

    def __call__(self, function, attribute, percentage):
        self.calls.append((function.value, attribute, percentage))
        return self.status


def construct(monkeypatch, *, status=0, shared=(31240, 55296, 0), capacity=233472):
    torch = pytest.importorskip("torch")
    properties = SimpleNamespace(major=9, minor=0, multi_processor_count=132,
                                 shared_memory_per_multiprocessor=capacity)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: properties)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: object())
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: object())
    setter = CarveoutSetter(status)
    monkeypatch.setattr(ctypes, "CDLL", lambda name: SimpleNamespace(cuFuncSetAttribute=setter))
    parents = ((), (), ("left", "right"))
    specs = tuple(dict(symbol=name, depends_on=dep, stream_index=stream, external_writes=(),
                       grid=(128, 1, 1), requires_grid_wide_sync=True)
                  for name, dep, stream in zip(("left", "right", "join"), parents, (0, 1, 0)))
    prepared = tuple(SimpleNamespace(compiled_kernel=SimpleNamespace(function=0x1000+i),
        contract=ResourceContract(4, 1), resource_report=dict(num_warps=4,
        registers_per_thread=32, shared_memory_bytes=n)) for i, n in enumerate(shared))
    return setter, lambda: ProgramDAG(specs, prepared, device="cuda:0", target="nvidia-sm90")


def test_dag_prepare_aligns_cache_preferences_on_both_sides_of_48k_threshold(monkeypatch):
    setter, build = construct(monkeypatch)
    dag = build()
    # Native loading gives >48KiB kernels PREFER_SHARED, but leaves smaller
    # entries at their default. Every entry (including the zero-shared join)
    # must share a preference before any graph capture or enqueue traversal.
    # 88832 joint bytes / 233472 physical bytes rounds upward to 39%.
    assert setter.calls == [(0x1000, 9, 39), (0x1001, 9, 39), (0x1002, 9, 39)]
    assert setter.argtypes == (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
    assert setter.restype is ctypes.c_int
    assert dag.resource_report["streams"] == 2
    assert dag.resource_report["shared_carveout_percent"] == 39


def test_dag_does_not_change_function_configuration_before_joint_admission(monkeypatch):
    setter, build = construct(monkeypatch, shared=(120000, 120000, 0))
    with pytest.raises(RuntimeContractError, match="joint SM90 residency"):
        build()
    assert setter.calls == []


def test_dag_rejects_failed_cache_configuration_without_silent_fallback(monkeypatch):
    setter, build = construct(monkeypatch, status=1)
    with pytest.raises(RuntimeContractError, match="cuFuncSetAttribute.*1"):
        build()
    assert setter.calls == [(0x1000, 9, 39)]


@pytest.mark.parametrize("capacity", (0, 1024))
def test_dag_rejects_unknown_or_insufficient_physical_shared_capacity(monkeypatch, capacity):
    setter, build = construct(monkeypatch, capacity=capacity)
    with pytest.raises(RuntimeContractError, match="physical shared"):
        build()
    assert setter.calls == []


def test_dag_reports_missing_driver_without_silent_fallback(monkeypatch):
    _, build = construct(monkeypatch)

    def missing(name):
        raise OSError("missing CUDA driver")

    monkeypatch.setattr(ctypes, "CDLL", missing)
    with pytest.raises(RuntimeContractError, match="CUDA driver"):
        build()
