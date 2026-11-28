# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.bufferization import AllocationStrategy, MemorySpace
from triton.flagmega.passes.tir.bufferize import BufferLifetime, SATBufferAllocator


def _workspace(maximum_bytes=4096):
    return MemorySpace(
        "workspace",
        "device",
        64,
        maximum_bytes,
        AllocationStrategy.SAT,
    )


def test_sat_allocator_reuses_disjoint_lifetimes_and_keeps_live_values_apart():
    result = SATBufferAllocator().allocate((
        BufferLifetime("a", 128, 64, 0, 1),
        BufferLifetime("b", 256, 64, 1, 2),
        BufferLifetime("c", 128, 64, 2, 3),
    ), _workspace())

    offsets = result.offset_map
    assert offsets["a"] == offsets["c"]
    assert offsets["a"] != offsets["b"]
    assert result.pool_bytes == 384


def test_sat_allocator_finds_a_layout_below_linear_high_water_mark():
    result = SATBufferAllocator().allocate((
        BufferLifetime("long", 192, 64, 0, 4),
        BufferLifetime("early", 320, 64, 0, 1),
        BufferLifetime("late", 320, 64, 2, 4),
    ), _workspace())

    assert result.offset_map["early"] == result.offset_map["late"]
    assert result.pool_bytes == 512


def test_sat_allocator_honors_each_buffer_alignment():
    result = SATBufferAllocator().allocate((
        BufferLifetime("small", 32, 64, 0, 2),
        BufferLifetime("wide", 96, 256, 0, 2),
    ), _workspace())

    assert result.offset_map["small"] % 64 == 0
    assert result.offset_map["wide"] % 256 == 0


def test_sat_allocator_reports_an_infeasible_capacity():
    with pytest.raises(IRVerificationError, match="SAT allocation.*failed"):
        SATBufferAllocator().allocate((
            BufferLifetime("a", 128, 64, 0, 1),
            BufferLifetime("b", 128, 64, 0, 1),
        ), _workspace(192))
