# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Formal raw participant credits preserve legacy elected arrivals."""
import inspect
from types import SimpleNamespace

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.backends.compiler import GPUTarget
from triton._filecheck import run_parser
from triton.experimental.tle.language.gpu import core, types


@triton.jit
def _public_participants():
    b = tle.gpu.alloc_barrier(arrive_count=256, init=tle.gpu.READY)
    tle.gpu.barrier_arrive(b, arrive_count=1, phaseIdx=0, participant_count=256)


@triton.jit
def _legacy_mode(MODE: tl.constexpr):
    b = tle.gpu.alloc_barrier(arrive_count=256, init=tle.gpu.READY)
    if MODE == 0:
        tle.gpu.barrier_arrive(b, 1, 0)
    else:
        tle.gpu.barrier_arrive(b, 1, 0, participant_count=None)


@triton.jit
def _legacy_named():
    b = tle.gpu.alloc_barrier(arrive_count=256)
    tle.gpu.barrier_arrive(b)


def _parse(fn, **kwargs):
    return run_parser(fn, kwargs=dict(kwargs, num_warps=8),
                      target=GPUTarget("cuda", 90, 32)).str_nodebug()


def test_public_participant_count_reaches_formal_TLE_IR():
    text = _parse(_public_participants)
    assert "participant_count = 256 : i32" in text
    line = next(line for line in text.splitlines() if "tle.barrier.arrive" in line)
    assert 'backend = "mbarrier"' in line and "arrive_count = 1 : i32" in line


def test_omitted_and_none_preserve_old_elected_builder():
    lines = [next(line for line in _parse(_legacy_mode, MODE=m).splitlines()
                  if "tle.barrier.arrive" in line).strip() for m in [0, 1]]
    assert lines[0] == lines[1]
    assert "participant_count" not in lines[0]
    assert 'backend = "named"' in _parse(_legacy_named)
    parameters = inspect.signature(core.barrier_arrive).parameters
    assert list(parameters)[:4] == ["barr", "arrive_count", "phaseIdx", "participant_count"]
    assert parameters["participant_count"].default is None


def _guard_slot(pending=256, expect=None, known=True):
    # Invalid requests must fail before any phase conversion/builder mutation.
    slot = object.__new__(types.barrier)
    slot.shape = [1]
    slot.arrive_count = pending
    slot.expect_bytes = expect
    slot.allocation_key = object() if known else None
    semantic = SimpleNamespace(builder=SimpleNamespace())
    return slot, semantic


@pytest.mark.parametrize("value", [False, True, 0, -1, 256.0, 2147483648,
                                   tl.constexpr(False), tl.constexpr(0), object.__new__(tl.tensor)])
def test_invalid_participant_width_rejected_before_builder(value):
    slot, semantic = _guard_slot()
    with pytest.raises(ValueError, match="positive compile-time int32"):
        core.barrier_arrive(slot, phaseIdx=0, participant_count=value, _semantic=semantic)


@pytest.mark.parametrize("pending,expect,known,arrive,phase,message", [
    (128, None, True, 1, 0, "known barrier allocation"),
    (256, None, False, 1, 0, "known barrier allocation"),
    (256, 128, True, 1, 0, "TX expect_bytes"),
    (256, None, True, 256, 0, "arrive_count = 1"),
    (256, None, True, 1, None, "explicit mbarrier phaseIdx"),
])
def test_wrong_credit_pending_TX_and_named_routes_rejected(pending, expect, known, arrive, phase, message):
    slot, semantic = _guard_slot(pending, expect, known)
    with pytest.raises(ValueError, match=message):
        core.barrier_arrive(slot, arrive_count=arrive, phaseIdx=phase,
                           participant_count=256, _semantic=semantic)


@pytest.mark.parametrize("backend", ["mthreads", "iluvatar", "common", "other"])
def test_unsupported_backend_rejected_before_builder(monkeypatch, backend):
    if backend == "mthreads":
        monkeypatch.setattr(core.mthreads_common, "enabled", lambda: True)
    elif backend == "iluvatar":
        monkeypatch.setattr(core.iluvatar_copy, "enabled", lambda: True)
    elif backend == "common":
        monkeypatch.setattr(core.tle_semantic, "COMMON_IR_ENABLED", True)
    else:
        from triton import _flagtree_backend
        monkeypatch.setattr(_flagtree_backend, "FLAGTREE_BACKEND", "unsupported")
    slot, semantic = _guard_slot()
    with pytest.raises(ValueError, match="NVIDIA mbarrier route"):
        core.barrier_arrive(slot, phaseIdx=0, participant_count=256, _semantic=semantic)
