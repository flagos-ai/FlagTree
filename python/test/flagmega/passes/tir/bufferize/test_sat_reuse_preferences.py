# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Barrier-sensitive reuse is a soft tie-break after the exact memory peak."""

import pytest
from itertools import product

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.bufferization import AllocationStrategy, MemorySpace
from triton.flagmega.passes.tir.bufferize import BufferLifetime, SATBufferAllocator


def _space():
    return MemorySpace("workspace", "device", 64, 4096, AllocationStrategy.SAT)


def _overlap(result, lhs, rhs, sizes):
    offsets = result.offset_map
    return offsets[lhs] < offsets[rhs] + sizes[rhs] and offsets[rhs] < offsets[lhs] + sizes[lhs]


def test_avoid_reuse_without_increasing_selected_high_water():
    lifetimes = (BufferLifetime("anchor", 256, 64, 0, 0),
                 BufferLifetime("stat", 64, 64, 1, 2),
                 BufferLifetime("output", 128, 64, 3, 4))
    sizes = {value.id: value.nbytes for value in lifetimes}
    allocator = SATBufferAllocator()
    baseline = allocator.allocate(lifetimes, _space())
    assert _overlap(baseline, "stat", "output", sizes)
    preferred = allocator.allocate(lifetimes, _space(), avoid_reuse=(("stat", "output"),))
    assert preferred.pool_bytes == baseline.pool_bytes == 256
    assert not _overlap(preferred, "stat", "output", sizes)
    assert preferred.reuse_conflicts == ()
    assert preferred.status == "OPTIMAL"


def test_reuse_preference_never_disables_memory_reuse_or_grows_peak():
    lifetimes = (BufferLifetime("a", 256, 64, 0, 1), BufferLifetime("b", 256, 64, 2, 3))
    result = SATBufferAllocator().allocate(lifetimes, _space(), avoid_reuse=(("a", "b"),))
    assert result.pool_bytes == 256
    assert result.offset_map == {"a": 0, "b": 0}
    assert result.reuse_conflicts == (("a", "b"),)


def test_reversed_duplicate_preferences_are_deterministic():
    lifetimes = (BufferLifetime("a", 128, 64, 0, 0), BufferLifetime("b", 64, 64, 1, 1))
    allocator = SATBufferAllocator()
    first = allocator.allocate(lifetimes, _space(), avoid_reuse=(("a", "b"),))
    second = allocator.allocate(lifetimes, _space(), avoid_reuse=(("b", "a"), ("a", "b")))
    assert first == second


@pytest.mark.parametrize("preferences", ((("missing", "a"),), (("a", "a"),), (("a",),)))
def test_reuse_preferences_must_name_two_existing_distinct_allocations(preferences):
    with pytest.raises(IRVerificationError, match="reuse"):
        SATBufferAllocator().allocate((BufferLifetime("a", 64, 64, 0, 0),), _space(), avoid_reuse=preferences)


def test_zero_byte_allocations_never_create_reuse_conflicts():
    result = SATBufferAllocator().allocate(
        (BufferLifetime("empty", 0, 64, 0, 0), BufferLifetime("value", 64, 64, 1, 1)),
        _space(), avoid_reuse=(("empty", "value"),),
    )
    assert result.pool_bytes == 64
    assert result.reuse_conflicts == ()


@pytest.mark.parametrize("windows", (
    ((0, 0), (1, 2), (3, 4)),
    ((0, 3), (0, 1), (2, 3)),
    ((0, 1), (1, 2), (2, 3)),
    ((0, 2), (1, 3), (2, 4)),
))
def test_sat_objectives_match_exhaustive_small_placement(windows):
    names = ("a", "b", "c")
    sizes = (128, 64, 64)
    lifetimes = tuple(BufferLifetime(name, size, 64, *window)
                      for name, size, window in zip(names, sizes, windows))
    pairs = (("a", "b"), ("b", "c"))
    possibilities = []
    for offsets in product(range(0, sum(sizes), 64), repeat=3):
        def overlaps(i, j):
            return offsets[i] < offsets[j] + sizes[j] and offsets[j] < offsets[i] + sizes[i]
        if any(overlaps(i, j) and windows[i][0] <= windows[j][1] and windows[j][0] <= windows[i][1]
               for i in range(3) for j in range(i)):
            continue
        possibilities.append((max(offset + size for offset, size in zip(offsets, sizes)),
                              int(overlaps(0, 1)) + int(overlaps(1, 2)), sum(offsets)))
    result = SATBufferAllocator().allocate(lifetimes, _space(), avoid_reuse=pairs)
    actual = (result.pool_bytes, len(result.reuse_conflicts), sum(result.offset_map.values()))
    assert actual == min(possibilities)
