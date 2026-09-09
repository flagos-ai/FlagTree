# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Deterministic address-ordered first fit over inclusive physical lifetimes."""

from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.ir.bufferization import MemorySpace
from .allocation import AllocationResult, BufferLifetime, align_up, reuse_conflicts, validate_problem, verify_allocation


def first_fit_placement(lifetimes: tuple[BufferLifetime, ...], space: MemorySpace) -> tuple[dict[str, int], int]:
    """Unbounded seed layout; the public allocator separately enforces capacity.

    Removing expired intervals coalesces their holes implicitly. At equal start
    times input order is stable, independent of symbol names. A value ending at
    a consumer's start is still live at that point.
    """
    offsets = {}
    active = []
    peak = 0
    for value in sorted(lifetimes, key=lambda item: item.live_start):
        if not value.nbytes:
            offsets[value.id] = 0
            continue
        active = [other for other in active if other.live_end >= value.live_start]
        alignment = max(value.alignment, space.granularity)
        offset = 0
        for other in active:
            if offset + value.nbytes <= offsets[other.id]:
                break
            offset = align_up(offsets[other.id] + other.nbytes, alignment)
        offsets[value.id] = offset
        peak = max(peak, offset + value.nbytes)
        active.append(value)
        active.sort(key=lambda item: offsets[item.id])
    return offsets, peak


class FirstFitBufferAllocator:
    name = "first-fit/lifetime"

    def allocate(self, lifetimes: tuple[BufferLifetime, ...], memory_space: MemorySpace, *,
                 avoid_reuse: tuple[tuple[str, str], ...] = ()) -> AllocationResult:
        pairs = validate_problem(lifetimes, memory_space, avoid_reuse)
        offsets, peak = first_fit_placement(lifetimes, memory_space)
        try:
            pool_bytes = memory_space.allocation_bytes(peak)
        except IRSchemaError as error:
            raise IRVerificationError(f"First-fit allocation exceeds {memory_space.name!r} capacity; "
                                      "select bufferize_opt_level='optimized' to search with SAT.") from error
        return verify_allocation(
            lifetimes, memory_space,
            AllocationResult(
                tuple((value.id, offsets[value.id]) for value in lifetimes),
                pool_bytes,
                "FEASIBLE",
                reuse_conflicts(lifetimes, offsets, pairs),
            ))


__all__ = ["FirstFitBufferAllocator"]
