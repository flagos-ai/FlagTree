# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Reuse exact physical allocation problems within one bufferization run."""

from dataclasses import replace

from .allocation import validate_problem, verify_allocation
from .first_fit_allocator import FirstFitBufferAllocator
from .sat_allocator import SATBufferAllocator


class AllocationSession:

    def __init__(self, options):
        self.optimization_level = options.optimization_level
        self.solver_time_seconds = options.solver_time_seconds
        self.allocator = (FirstFitBufferAllocator() if self.optimization_level == "fast" else SATBufferAllocator(
            maximum_time_seconds=self.solver_time_seconds))
        self.name = self.allocator.name
        self._results = {}
        self.hits = 0
        self.misses = 0

    def allocate(self, lifetimes, memory_space, *, avoid_reuse=()):
        pairs = validate_problem(lifetimes, memory_space, avoid_reuse)
        indexes = {value.id: index for index, value in enumerate(lifetimes)}
        canonical = tuple(replace(value, id=str(index)) for index, value in enumerate(lifetimes))
        canonical_pairs = tuple(sorted(tuple(sorted((str(indexes[a]), str(indexes[b])))) for a, b in pairs))
        key = (memory_space, canonical, canonical_pairs)
        if key not in self._results:
            self._results[key] = self.allocator.allocate(canonical, memory_space, avoid_reuse=canonical_pairs)
            self.misses += 1
        else:
            self.hits += 1
        result = self._results[key]
        names = {str(index): value.id for index, value in enumerate(lifetimes)}
        mapped = replace(
            result, offsets=tuple((names[name], offset) for name, offset in result.offsets),
            reuse_conflicts=tuple(sorted(tuple(sorted((names[a], names[b]))) for a, b in result.reuse_conflicts)))
        return verify_allocation(lifetimes, memory_space, mapped)


__all__ = ["AllocationSession"]
