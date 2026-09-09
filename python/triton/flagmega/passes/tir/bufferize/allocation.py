# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Allocator-independent finite lifetime problems and verified placements."""

from dataclasses import dataclass
from typing import Protocol

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.bufferization import AllocationPolicy, MemorySpace


@dataclass(frozen=True)
class BufferLifetime:
    id: str
    nbytes: int
    alignment: int
    live_start: int
    live_end: int
    role: str = "workspace"

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise IRVerificationError("An allocation requires a non-empty id.")
        if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (self.nbytes, self.alignment, self.live_start, self.live_end)):
            raise IRVerificationError("Allocation sizes, alignment and lifetime endpoints must be integers.")
        if self.nbytes < 0:
            raise IRVerificationError(f"Allocation {self.id!r} has negative size.")
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise IRVerificationError(f"Allocation {self.id!r} alignment must be a positive power of two.")
        if self.live_start < 0 or self.live_end < self.live_start:
            raise IRVerificationError(f"Allocation {self.id!r} has an invalid lifetime.")


@dataclass(frozen=True)
class AllocationObjective:
    name: str
    status: str
    value: int
    best_bound: float | None = None

    def to_data(self):
        return {"name": self.name, "status": self.status, "value": self.value, "best_bound": self.best_bound}


@dataclass(frozen=True)
class AllocationResult:
    offsets: tuple[tuple[str, int], ...]
    pool_bytes: int
    status: str
    reuse_conflicts: tuple[tuple[str, str], ...] = ()
    objectives: tuple[AllocationObjective, ...] = ()

    @property
    def offset_map(self):
        return dict(self.offsets)


class BufferAllocator(Protocol):
    name: str

    def allocate(self, lifetimes: tuple[BufferLifetime, ...], memory_space: MemorySpace, *,
                 avoid_reuse: tuple[tuple[str, str], ...] = ()) -> AllocationResult:
        ...


def validate_problem(lifetimes, space, avoid_reuse=()):
    if not space.supports_lifetime_reuse:
        raise IRVerificationError(f"Memory space {space.name!r} does not support lifetime allocation.")
    ids = {value.id for value in lifetimes}
    if len(ids) != len(lifetimes):
        raise IRVerificationError("Allocation ids must be unique.")
    pairs = set()
    for pair in avoid_reuse:
        if len(pair) != 2 or pair[0] == pair[1] or any(value not in ids for value in pair):
            raise IRVerificationError("Allocation reuse preferences must name two existing distinct allocations.")
        pairs.add(tuple(sorted(pair)))
    return tuple(sorted(pairs))


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def usable_capacity(space):
    """Allocation-policy rounding must not exceed the physical capacity."""
    if space.maximum_bytes < space.granularity:
        return 0
    if space.allocation_policy is AllocationPolicy.POWER_OF_TWO:
        return 1 << (space.maximum_bytes.bit_length() - 1)
    return space.maximum_bytes // space.granularity * space.granularity


def reuse_conflicts(lifetimes, offsets, pairs):
    sizes = {value.id: value.nbytes for value in lifetimes}
    return tuple((left, right)
                 for left, right in pairs
                 if sizes[left] and sizes[right] and offsets[left] < offsets[right] +
                 sizes[right] and offsets[right] < offsets[left] + sizes[left])


def verify_allocation(lifetimes, space, result):
    """Validate every allocator's result before it can become a MemSpan."""
    offsets = result.offset_map
    if len(offsets) != len(result.offsets) or set(offsets) != {value.id for value in lifetimes}:
        raise IRVerificationError("Allocation result must cover exactly the physical lifetimes.")
    peak = 0
    active = []
    for value in sorted(lifetimes, key=lambda item: item.live_start):
        offset = offsets[value.id]
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise IRVerificationError(f"Allocation {value.id!r} has an invalid offset.")
        if offset % max(value.alignment, space.granularity):
            raise IRVerificationError(f"Allocation {value.id!r} is not aligned.")
        if not value.nbytes:
            if offset:
                raise IRVerificationError(f"Zero-byte allocation {value.id!r} must use offset zero.")
            continue
        peak = max(peak, offset + value.nbytes)
        active = [other for other in active if other.live_end >= value.live_start]
        for other in active:
            if offset < offsets[other.id] + other.nbytes and offsets[other.id] < offset + value.nbytes:
                raise IRVerificationError(f"Live allocations {value.id!r} and {other.id!r} overlap.")
        active.append(value)
    if space.allocation_bytes(peak) != result.pool_bytes:
        raise IRVerificationError("Allocation result has an inconsistent pool size.")
    return result


__all__ = ["AllocationObjective", "AllocationResult", "BufferAllocator", "BufferLifetime"]
