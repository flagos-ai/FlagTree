# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Linked device communication must respect a warp-specialized caller."""

import re

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit(noinline=True)
def _barrier_helper(mem, comm, KIND: tl.constexpr):
    if KIND == 0:
        tle.distributed_barrier(device_dptr=(mem, comm), space="device", barrier_kind="arrive", index=0,
                                barrier_count=1)
    elif KIND == 1:
        tle.distributed_barrier(device_dptr=(mem, comm), space="device", barrier_kind="wait", index=0, barrier_count=1)
    else:
        tle.distributed_barrier(device_dptr=(mem, comm), space="device", barrier_kind="sync", index=0, barrier_count=1)


@triton.jit
def _consumer(out, mem, comm, KIND: tl.constexpr, NESTED: tl.constexpr):
    if NESTED:
        _barrier_helper(mem, comm, KIND)
    elif KIND == 0:
        tle.distributed_barrier(device_dptr=(mem, comm), space="device", barrier_kind="arrive", index=0,
                                barrier_count=1)
    elif KIND == 1:
        tle.distributed_barrier(device_dptr=(mem, comm), space="device", barrier_kind="wait", index=0, barrier_count=1)
    else:
        tle.distributed_barrier(device_dptr=(mem, comm), space="device", barrier_kind="sync", index=0, barrier_count=1)
    tl.store(out + tl.arange(0, 128), 1)


@triton.jit
def _producer(out):
    tl.store(out + 128 + tl.arange(0, 128), 2)


@triton.jit
def _entry(out, mem, comm, KIND: tl.constexpr, NESTED: tl.constexpr):
    tle.gpu.warp_specialize([(_consumer, (out, mem, comm, KIND, NESTED)), (_producer, (out, ))], [4], [24])


@pytest.mark.parametrize("kind", (0, 1, 2))
@pytest.mark.parametrize("nested", (False, True))
def test_linked_flagcx_barrier_has_no_whole_cta_sync(kind, nested, monkeypatch):
    # Compile only. No CUDA context or representative handle is dereferenced.
    from triton import _flagtree_backend
    monkeypatch.setattr(_flagtree_backend, "FLAGTREE_BACKEND", "nvidia")
    compiled = triton.compile(
        ASTSource(_entry, signature={"out": "*i32", "mem": "i64", "comm": "i64"},
                  constexprs={"KIND": kind, "NESTED": nested}), target=GPUTarget("cuda", 90, 32),
        options={"num_warps": 8})
    assert not re.search(r"\bbar\.sync\s+0\s*;", compiled.asm["ptx"]), \
        "FlagCX block cooperation still waits for the producer's threads"
