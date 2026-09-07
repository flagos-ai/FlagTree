# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Exact lifetime-aware allocation using OR-Tools CP-SAT.

The model follows nncase's ``SATBufferScheduler``: every buffer is a rectangle
whose X interval is its fixed lifetime and whose Y interval is its physical
address range.  ``NoOverlap2D`` therefore prohibits two simultaneously-live
buffers from occupying intersecting bytes.
"""

from __future__ import annotations

from dataclasses import dataclass

from ortools.sat.python import cp_model

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.bufferization import AllocationStrategy, MemorySpace


@dataclass(frozen=True)
class BufferLifetime:
    id: str
    nbytes: int
    alignment: int
    live_start: int
    live_end: int
    role: str = "workspace"

    def __post_init__(self) -> None:
        if not self.id:
            raise IRVerificationError("A SAT allocation requires a non-empty id.")
        if self.nbytes < 0:
            raise IRVerificationError(f"Allocation {self.id!r} has negative size.")
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise IRVerificationError(
                f"Allocation {self.id!r} alignment must be a positive power of two."
            )
        if self.live_start < 0 or self.live_end < self.live_start:
            raise IRVerificationError(f"Allocation {self.id!r} has an invalid lifetime.")


@dataclass(frozen=True)
class SATAllocationResult:
    offsets: tuple[tuple[str, int], ...]
    pool_bytes: int
    status: str
    reuse_conflicts: tuple[tuple[str, str], ...] = ()

    @property
    def offset_map(self) -> dict[str, int]:
        return dict(self.offsets)


class SATBufferAllocator:
    """Place finite lifetimes in one target memory space."""

    name = "ortools-cp-sat/no-overlap-2d"

    def __init__(self, *, maximum_time_seconds: float = 30.0) -> None:
        if maximum_time_seconds <= 0:
            raise ValueError("maximum_time_seconds must be positive.")
        self.maximum_time_seconds = float(maximum_time_seconds)

    def allocate(
        self,
        lifetimes: tuple[BufferLifetime, ...],
        memory_space: MemorySpace,
        *,
        avoid_reuse: tuple[tuple[str, str], ...] = (),
    ) -> SATAllocationResult:
        if memory_space.strategy is not AllocationStrategy.SAT:
            raise IRVerificationError(
                f"Memory space {memory_space.name!r} does not use SAT allocation."
            )
        ids = [value.id for value in lifetimes]
        if len(ids) != len(set(ids)):
            raise IRVerificationError("SAT allocation ids must be unique.")
        pairs = set()
        for pair in avoid_reuse:
            if len(pair) != 2 or pair[0] == pair[1] or any(value not in ids for value in pair):
                raise IRVerificationError(
                    "SAT reuse preferences must name two existing distinct allocations."
                )
            pairs.add(tuple(sorted(pair)))
        pairs = tuple(sorted(pairs))
        if not lifetimes:
            return SATAllocationResult((), 0, "OPTIMAL")

        nonempty = tuple(value for value in lifetimes if value.nbytes)
        if not nonempty:
            return SATAllocationResult(tuple((value.id, 0) for value in lifetimes), 0, "OPTIMAL")

        # A packed linear layout is always a legal upper bound.  It also keeps
        # every CP-SAT integer domain finite without relying on target infinity.
        upper_bound = 0
        for value in nonempty:
            alignment = max(memory_space.granularity, value.alignment)
            upper_bound = _align_up(upper_bound, alignment) + value.nbytes
        if upper_bound > memory_space.maximum_bytes:
            # Overlapping lifetimes can still fit below the linear bound, so use
            # the target limit as the domain and let CP-SAT decide feasibility.
            upper_bound = memory_space.maximum_bytes

        model = cp_model.CpModel()
        pool_end = model.new_int_var(0, upper_bound, "pool_end")
        starts: dict[str, cp_model.IntVar] = {}
        x_intervals = []
        y_intervals = []
        for ordinal, value in enumerate(nonempty):
            alignment = max(memory_space.granularity, value.alignment)
            latest = upper_bound - value.nbytes
            if latest < 0:
                raise IRVerificationError(
                    f"Allocation {value.id!r} ({value.nbytes} bytes) exceeds memory space "
                    f"{memory_space.name!r} ({memory_space.maximum_bytes} bytes)."
                )
            quotient = model.new_int_var(0, latest // alignment, f"q_{ordinal}")
            start = model.new_int_var(0, latest, f"offset_{ordinal}")
            model.add(start == quotient * alignment)
            end = model.new_int_var(value.nbytes, upper_bound, f"end_{ordinal}")
            model.add(end == start + value.nbytes)
            model.add(pool_end >= end)
            starts[value.id] = start

            # Lifetimes are inclusive in IR metadata.  Converting to half-open
            # intervals with +1 preserves the rule that a consumer and its
            # input overlap at the consumer's execution point.
            duration = value.live_end - value.live_start + 1
            x_intervals.append(model.new_fixed_size_interval_var(
                value.live_start, duration, f"time_{ordinal}"
            ))
            y_intervals.append(model.new_interval_var(
                start, value.nbytes, end, f"address_{ordinal}"
            ))
        model.add_no_overlap_2d(x_intervals, y_intervals)

        solver = self._solver()
        model.minimize(pool_end)
        status = solver.solve(model)
        self._require_solution(status, solver, memory_space)
        high_water_status = solver.status_name(status)
        selected_pool_end = solver.value(pool_end)

        # A second objective makes layouts at the selected high-water mark
        # reproducible and tends to place long-lived values low in the pool.
        model.add(pool_end == selected_pool_end)
        # Reusing particular allocations can force an otherwise unnecessary
        # inter-owner barrier. Minimize those proven conflicts at the already
        # selected high-water mark, never by abandoning reuse or growing the
        # pool. This is a soft preference: unavoidable conflicts remain legal
        # and are reported for the synchronization planner to handle.
        sizes = {value.id: value.nbytes for value in nonempty}
        reuse = {}
        for ordinal, (left, right) in enumerate(pairs):
            if left not in sizes or right not in sizes:
                continue
            overlaps = model.new_bool_var(f"reuse_{ordinal}")
            left_before = model.new_bool_var(f"reuse_left_before_{ordinal}")
            right_before = model.new_bool_var(f"reuse_right_before_{ordinal}")
            model.add(starts[left] + sizes[left] <= starts[right]).only_enforce_if(left_before)
            model.add(starts[right] + sizes[right] <= starts[left]).only_enforce_if(right_before)
            model.add_bool_or((left_before, right_before)).only_enforce_if(overlaps.Not())
            model.add(starts[left] + sizes[left] > starts[right]).only_enforce_if(overlaps)
            model.add(starts[right] + sizes[right] > starts[left]).only_enforce_if(overlaps)
            reuse[(left, right)] = overlaps
        reuse_status = "OPTIMAL"
        if reuse:
            conflict_count = sum(reuse.values())
            model.minimize(conflict_count)
            solver = self._solver()
            status = solver.solve(model)
            self._require_solution(status, solver, memory_space)
            reuse_status = solver.status_name(status)
            model.add(conflict_count == solver.value(conflict_count))
        model.minimize(sum(starts.values()))
        solver = self._solver()
        status = solver.solve(model)
        self._require_solution(status, solver, memory_space)
        allocated_bytes = memory_space.allocation_bytes(selected_pool_end)
        offsets = tuple(
            (value.id, 0 if value.nbytes == 0 else solver.value(starts[value.id]))
            for value in lifetimes
        )
        tie_break_status = solver.status_name(status)
        combined_status = (
            "OPTIMAL"
            if high_water_status == reuse_status == tie_break_status == "OPTIMAL"
            else f"high-water:{high_water_status};reuse:{reuse_status};tie-break:{tie_break_status}"
        )
        return SATAllocationResult(
            offsets, allocated_bytes, combined_status,
            tuple(pair for pair, overlaps in reuse.items() if solver.value(overlaps)),
        )

    def _solver(self) -> cp_model.CpSolver:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.maximum_time_seconds
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        return solver

    @staticmethod
    def _require_solution(
        status: int,
        solver: cp_model.CpSolver,
        memory_space: MemorySpace,
    ) -> None:
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise IRVerificationError(
                f"SAT allocation for memory space {memory_space.name!r} failed with "
                f"status {solver.status_name(status)} and capacity "
                f"{memory_space.maximum_bytes} bytes."
            )


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


__all__ = ["BufferLifetime", "SATAllocationResult", "SATBufferAllocator"]
