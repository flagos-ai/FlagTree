# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nncase-shaped high-water SAT allocation, optionally avoiding reuse hazards."""

from __future__ import annotations

from math import isfinite
from time import perf_counter

from ortools.sat.python import cp_model

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.bufferization import MemorySpace
from .allocation import (
    AllocationObjective,
    AllocationResult,
    BufferLifetime,
    reuse_conflicts,
    usable_capacity,
    validate_problem,
    verify_allocation,
)
from .first_fit_allocator import first_fit_placement

# Compatibility import; both allocation algorithms return the same contract.
SATAllocationResult = AllocationResult


class SATBufferAllocator:
    name = "ortools-cp-sat/no-overlap-2d"

    def __init__(self, *, maximum_time_seconds: float = 30.0) -> None:
        if not isfinite(maximum_time_seconds) or maximum_time_seconds <= 0:
            raise ValueError("maximum_time_seconds must be finite and positive.")
        self.maximum_time_seconds = float(maximum_time_seconds)

    def allocate(self, lifetimes: tuple[BufferLifetime, ...], memory_space: MemorySpace, *,
                 avoid_reuse: tuple[tuple[str, str], ...] = ()) -> AllocationResult:
        pairs = validate_problem(lifetimes, memory_space, avoid_reuse)
        deadline = perf_counter() + self.maximum_time_seconds
        nonempty = tuple(value for value in lifetimes if value.nbytes)
        if not nonempty:
            return AllocationResult(tuple((value.id, 0) for value in lifetimes), 0, "OPTIMAL")

        seed, seed_peak = first_fit_placement(nonempty, memory_space)
        capacity = usable_capacity(memory_space)
        upper_bound = min(seed_peak, capacity)
        events = {}
        for value in nonempty:
            events[value.live_start] = events.get(value.live_start, 0) + value.nbytes
            events[value.live_end + 1] = events.get(value.live_end + 1, 0) - value.nbytes
        live_bytes = lower_bound = 0
        for point in sorted(events):
            live_bytes += events[point]
            lower_bound = max(lower_bound, live_bytes)
        if lower_bound > upper_bound:
            raise IRVerificationError(f"SAT allocation for memory space {memory_space.name!r} failed: "
                                      f"live-byte lower bound {lower_bound} exceeds capacity {capacity}.")

        model = cp_model.CpModel()
        pool_end = model.new_int_var(lower_bound, upper_bound, "pool_end")
        variables = {}
        x_intervals, y_intervals = [], []
        for ordinal, value in enumerate(nonempty):
            alignment = max(memory_space.granularity, value.alignment)
            latest = upper_bound - value.nbytes
            quotient = model.new_int_var(0, latest // alignment, f"q_{ordinal}")
            start = model.new_int_var(0, latest, f"offset_{ordinal}")
            end = model.new_int_var(value.nbytes, upper_bound, f"end_{ordinal}")
            model.add(start == quotient * alignment)
            model.add(end == start + value.nbytes)
            variables[value.id] = (quotient, start, end, alignment, value.nbytes)
            x_intervals.append(
                model.new_fixed_size_interval_var(value.live_start, value.live_end - value.live_start + 1,
                                                  f"time_{ordinal}"))
            y_intervals.append(model.new_interval_var(start, value.nbytes, end, f"address_{ordinal}"))
        model.add_no_overlap_2d(x_intervals, y_intervals)
        model.add_max_equality(pool_end, [value[2] for value in variables.values()])

        def add_hint(offsets, peak):
            model.clear_hints()
            model.add_hint(pool_end, peak)
            for name, (quotient, start, end, alignment, size) in variables.items():
                model.add_hint(quotient, offsets[name] // alignment)
                model.add_hint(start, offsets[name])
                model.add_hint(end, offsets[name] + size)

        if seed_peak <= capacity:
            add_hint(seed, seed_peak)
        model.minimize(pool_end)
        solver = self._solver(deadline)
        status = solver.solve(model)
        self._require_solution(status, solver, memory_space)
        high_water_status = solver.status_name(status)
        selected_peak = solver.value(pool_end)
        offsets = {name: solver.value(value[1]) for name, value in variables.items()}
        objectives = [AllocationObjective("high_water", high_water_status, selected_peak, solver.best_objective_bound)]
        model.add(pool_end == selected_peak)

        # Unlike sum(addresses), this objective corresponds to proven
        # inter-owner hazards. All objectives share one allocation budget.
        nonempty_pairs = tuple(pair for pair in pairs if all(name in variables for name in pair))
        reuse_status = "OPTIMAL"
        if nonempty_pairs:
            add_hint(offsets, selected_peak)
            overlap_vars = []
            for ordinal, (left, right) in enumerate(nonempty_pairs):
                _, left_start, left_end, _, left_size = variables[left]
                _, right_start, right_end, _, right_size = variables[right]
                overlap = model.new_bool_var(f"reuse_{ordinal}")
                left_before = model.new_bool_var(f"reuse_left_before_{ordinal}")
                right_before = model.new_bool_var(f"reuse_right_before_{ordinal}")
                model.add(left_end <= right_start).only_enforce_if(left_before)
                model.add(right_end <= left_start).only_enforce_if(right_before)
                model.add_bool_or((left_before, right_before)).only_enforce_if(overlap.Not())
                model.add(left_end > right_start).only_enforce_if(overlap)
                model.add(right_end > left_start).only_enforce_if(overlap)
                before_left = offsets[left] + left_size <= offsets[right]
                before_right = offsets[right] + right_size <= offsets[left]
                model.add_hint(left_before, int(before_left))
                model.add_hint(right_before, int(before_right))
                model.add_hint(overlap, int(not before_left and not before_right))
                overlap_vars.append(overlap)
            model.minimize(sum(overlap_vars))
            bound = None
            if perf_counter() < deadline:
                solver = self._solver(deadline)
                status = solver.solve(model)
                reuse_status = solver.status_name(status)
                if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    offsets = {name: solver.value(value[1]) for name, value in variables.items()}
                    bound = solver.best_objective_bound
                elif status != cp_model.UNKNOWN:
                    self._require_solution(status, solver, memory_space)
                # UNKNOWN retains the already proven feasible SAT placement,
                # with incomplete optimization explicitly recorded below.
            else:
                reuse_status = "NOT_RUN_BUDGET"
            conflicts = reuse_conflicts(nonempty, offsets, nonempty_pairs)
            objectives.append(AllocationObjective("reuse_conflicts", reuse_status, len(conflicts), bound))
        offsets.update((value.id, 0) for value in lifetimes if not value.nbytes)
        combined = ("OPTIMAL" if high_water_status == reuse_status == "OPTIMAL" else
                    f"high-water:{high_water_status};reuse:{reuse_status}")
        return verify_allocation(
            lifetimes, memory_space,
            AllocationResult(
                tuple((value.id, offsets[value.id]) for value in lifetimes),
                memory_space.allocation_bytes(selected_peak),
                combined,
                reuse_conflicts(lifetimes, offsets, pairs),
                tuple(objectives),
            ))

    def _solver(self, deadline):
        remaining = deadline - perf_counter()
        if remaining <= 0:
            raise IRVerificationError("SAT allocation budget exhausted before a feasible solution was available.")
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = remaining
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        return solver

    @staticmethod
    def _require_solution(status, solver, memory_space):
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise IRVerificationError(
                f"SAT allocation for memory space {memory_space.name!r} failed with "
                f"status {solver.status_name(status)} and capacity {memory_space.maximum_bytes} bytes.")


__all__ = ["BufferLifetime", "SATAllocationResult", "SATBufferAllocator"]
