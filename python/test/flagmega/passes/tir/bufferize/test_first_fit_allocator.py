# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""First-fit changes placement quality, never physical safety contracts."""

from dataclasses import replace

import pytest

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.bufferization import AllocationPolicy, AllocationStrategy, MemorySpace
from triton.flagmega.passes.tir.bufferize import BufferLifetime, FirstFitBufferAllocator, SATBufferAllocator
from triton.flagmega.passes.tir.bufferize.allocation import AllocationResult, verify_allocation


def space(capacity=4096, policy=AllocationPolicy.GRANULARITY_ALIGNED):
    return MemorySpace("workspace", "device", 64, capacity, AllocationStrategy.REUSE, policy)


def test_first_fit_reuses_coalesced_holes_and_inclusive_lifetimes():
    lifetimes = (
        BufferLifetime("a", 128, 64, 0, 1),
        BufferLifetime("b", 128, 64, 0, 1),
        BufferLifetime("c", 64, 64, 0, 4),
        BufferLifetime("d", 256, 64, 2, 3),
        BufferLifetime("e", 64, 64, 3, 4),
    )
    result = FirstFitBufferAllocator().allocate(lifetimes, space())
    assert result.offset_map == {"a": 0, "b": 128, "c": 256, "d": 0, "e": 320}
    assert result.pool_bytes == 384
    assert result.status == "FEASIBLE"


def test_first_fit_can_fill_alignment_padding():
    result = FirstFitBufferAllocator().allocate((
        BufferLifetime("a", 32, 64, 0, 1),
        BufferLifetime("b", 64, 256, 0, 1),
        BufferLifetime("c", 128, 64, 0, 1),
    ), space())
    assert result.offset_map == {"a": 0, "b": 256, "c": 64}
    assert result.pool_bytes == 320


@pytest.mark.parametrize("allocator", [FirstFitBufferAllocator, SATBufferAllocator])
def test_empty_and_zero_byte_allocations(allocator):
    assert allocator().allocate((), space()).pool_bytes == 0
    result = allocator().allocate((BufferLifetime("zero", 0, 256, 0, 1), ), space())
    assert result.offset_map == {"zero": 0}
    assert result.pool_bytes == 0


def test_fast_capacity_failure_is_explicit_and_optimized_can_fit():
    # First-fit puts the more aligned allocation second; SAT can reorder it.
    lifetimes = (BufferLifetime("small", 64, 64, 0, 1), BufferLifetime("wide", 128, 256, 0, 1))
    with pytest.raises(IRVerificationError, match="optimized"):
        FirstFitBufferAllocator().allocate(lifetimes, space(256))
    result = SATBufferAllocator().allocate(lifetimes, space(256))
    assert result.pool_bytes == 192
    assert result.offset_map == {"small": 128, "wide": 0}


@pytest.mark.parametrize("allocator", [FirstFitBufferAllocator, SATBufferAllocator])
def test_capacity_includes_power_of_two_rounding(allocator):
    lifetimes = (BufferLifetime("x", 192, 64, 0, 1), )
    with pytest.raises(IRVerificationError):
        allocator().allocate(lifetimes, space(224, AllocationPolicy.POWER_OF_TWO))
    assert allocator().allocate(lifetimes, space(256, AllocationPolicy.POWER_OF_TWO)).pool_bytes == 256


@pytest.mark.parametrize("allocator", [FirstFitBufferAllocator, SATBufferAllocator])
def test_shared_verifier_rejects_overlap_and_bad_alignment(allocator):
    lifetimes = (BufferLifetime("a", 64, 64, 0, 1), BufferLifetime("b", 64, 64, 1, 2))
    result = allocator().allocate(lifetimes, space())
    with pytest.raises(IRVerificationError, match="overlap"):
        verify_allocation(lifetimes, space(), replace(result, offsets=(("a", 0), ("b", 0))))
    with pytest.raises(IRVerificationError, match="aligned"):
        verify_allocation(lifetimes, space(), AllocationResult((("a", 1), ("b", 128)), 192, result.status))


@pytest.mark.parametrize("allocator", [FirstFitBufferAllocator, SATBufferAllocator])
def test_duplicate_ids_and_invalid_preferences_are_not_accepted(allocator):
    value = BufferLifetime("a", 64, 64, 0, 1)
    with pytest.raises(IRVerificationError, match="unique"):
        allocator().allocate((value, value), space())
    with pytest.raises(IRVerificationError, match="existing distinct"):
        allocator().allocate((value, ), space(), avoid_reuse=(("a", "missing"), ))


@pytest.mark.parametrize("allocator", [FirstFitBufferAllocator, SATBufferAllocator])
@pytest.mark.parametrize("policy", list(AllocationPolicy))
def test_capacity_smaller_than_granularity_rejects_nonempty_pool(allocator, policy):
    with pytest.raises(IRVerificationError):
        allocator().allocate((BufferLifetime("x", 8, 64, 0, 0), ), space(32, policy))
