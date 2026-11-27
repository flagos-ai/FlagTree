# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nncase-shaped high-water SAT allocation, optionally avoiding reuse hazards."""

from __future__ import annotations

from collections import Counter
from heapq import heappop, heappush
from math import isfinite
from time import perf_counter

from ortools.sat.python import cp_model

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.bufferization import MemorySpace
from .allocation import (
    AllocationObjective,
    AllocationResult,
    BufferLifetime,
    align_up,
    reuse_conflicts,
    usable_capacity,
    validate_problem,
    verify_allocation,
)
from .first_fit_allocator import first_fit_placement

# Compatibility import; both allocation algorithms return the same contract.
SATAllocationResult = AllocationResult


def _aligned_live_peak_lower_bound(lifetimes, granularity):
    # Aligned starts reserve ceil(size / granularity) whole address cells for
    # simultaneous roots. Only the last root's trailing padding can lie above
    # the raw byte high water. Larger per-root alignments can only raise it.
    events = {}
    for value in lifetimes:
        padded = align_up(value.nbytes, granularity)
        slack = padded - value.nbytes
        events.setdefault(value.live_start, []).append((1, padded, slack))
        events.setdefault(value.live_end + 1, []).append((-1, padded, slack))
    live_bytes = lower = 0
    slacks = Counter()
    maximum_slacks = []
    for point in sorted(events):
        for sign, padded, slack in events[point]:
            live_bytes += sign * padded
            slacks[slack] += sign
            if sign > 0:
                heappush(maximum_slacks, -slack)
        while maximum_slacks and not slacks[-maximum_slacks[0]]:
            heappop(maximum_slacks)
        lower = max(lower, live_bytes + (maximum_slacks[0] if maximum_slacks else 0))
    return lower


def _clique_bound_witness(lifetimes, space, lower_bound):
    """Try an exact placement for simultaneous roots, including tail padding.

    Only the final root can put its trailing address-cell slack above the raw
    high water. Larger alignments may create extra holes, so accept this witness
    only when it actually attains the independently proved live-peak bound.
    """
    if max(v.live_start for v in lifetimes) > min(v.live_end for v in lifetimes):
        return None
    last = max(range(len(lifetimes)), key=lambda i:
               align_up(lifetimes[i].nbytes, space.granularity) - lifetimes[i].nbytes)
    order = sorted((i for i in range(len(lifetimes)) if i != last),
                   key=lambda i: (-max(space.granularity, lifetimes[i].alignment), i))
    offsets, cursor = {}, 0
    for i in (*order, last):
        value = lifetimes[i]
        start = align_up(cursor, max(space.granularity, value.alignment))
        offsets[value.id] = start
        cursor = start + value.nbytes
    return offsets if cursor == lower_bound else None


def _disjoint_cliques(lifetimes, pairs, granularity):
    """Bound peak bytes when every requested reuse pair is disjoint.

    A clique of temporally conflicting roots plus zero-reuse pairs needs
    disjoint addresses. Greedy cliques need not be maximum to prove a bound.
    """
    positions = {value.id: index for index, value in enumerate(lifetimes)}
    adjacency = [0] * len(lifetimes)
    seeds = set()
    for left, right in pairs:
        i, j = positions[left], positions[right]
        adjacency[i] |= 1 << j
        adjacency[j] |= 1 << i
        seeds.update((i, j))
    for i, left in enumerate(lifetimes):
        for j in range(i):
            right = lifetimes[j]
            if left.live_start <= right.live_end and right.live_start <= left.live_end:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    weights = [align_up(value.nbytes, granularity) for value in lifetimes]
    slack = [weight - value.nbytes for value, weight in zip(lifetimes, weights)]
    cliques = {}
    for seed in sorted(seeds):
        for degree_first in (False, True):
            total, padding = weights[seed], slack[seed]
            members = 1 << seed
            remaining = adjacency[seed]
            while remaining:
                choices = []
                bits = remaining
                while bits:
                    bit = bits & -bits
                    choices.append(bit.bit_length() - 1)
                    bits ^= bit
                def score(index):
                    degree = (adjacency[index] & remaining).bit_count()
                    return ((degree, weights[index], -index) if degree_first
                            else (weights[index], degree, -index))
                selected = max(choices, key=score)
                total += weights[selected]
                members |= 1 << selected
                padding = max(padding, slack[selected])
                remaining &= adjacency[selected]
            pair_indices = tuple(index for index, (a, b) in enumerate(pairs)
                                 if members & (1 << positions[a]) and members & (1 << positions[b]))
            cliques[members] = total - padding, pair_indices
    return tuple(cliques.values())


def _disjoint_clique_lower_bound(lifetimes, pairs, granularity):
    return max((bound for bound, _ in _disjoint_cliques(lifetimes, pairs, granularity)), default=0)


def _reuse_clique_lower_bound(lifetimes, pairs, granularity, conflict_count):
    if conflict_count == 0:
        return _disjoint_clique_lower_bound(lifetimes, pairs, granularity)
    sizes = {value.id: align_up(value.nbytes, granularity) for value in lifetimes}
    lower = 0
    for bound, indices in _disjoint_cliques(lifetimes, pairs, granularity):
        savings = sorted((min(sizes[pairs[index][0]], sizes[pairs[index][1]]) for index in indices), reverse=True)
        lower = max(lower, bound - sum(savings[:conflict_count]))
    return lower


class SATBufferAllocator:
    name = "ortools-cp-sat/no-overlap-2d"

    def __init__(self, *, maximum_time_seconds: float = 30.0) -> None:
        if not isfinite(maximum_time_seconds) or maximum_time_seconds <= 0:
            raise ValueError("maximum_time_seconds must be finite and positive.")
        self.maximum_time_seconds = float(maximum_time_seconds)

    def allocate(self, lifetimes: tuple[BufferLifetime, ...], memory_space: MemorySpace, *,
                 avoid_reuse: tuple[tuple[str, str], ...] = (),
                 bytes_budget: int = 0) -> AllocationResult:
        if isinstance(bytes_budget, bool) or not isinstance(bytes_budget, int) or bytes_budget < 0:
            raise IRVerificationError(
                f"The reuse-avoidance byte budget must be a non-negative integer, got {bytes_budget!r}.")
        pairs = validate_problem(lifetimes, memory_space, avoid_reuse)
        deadline = perf_counter() + self.maximum_time_seconds
        nonempty = tuple(value for value in lifetimes if value.nbytes)
        if not nonempty:
            return AllocationResult(tuple((value.id, 0) for value in lifetimes), 0, "OPTIMAL")

        nonempty_ids = {value.id for value in nonempty}
        nonempty_pairs = tuple(pair for pair in pairs if all(name in nonempty_ids for name in pair))
        # Extra bytes serve the secondary reuse objective. Without an active
        # pair, expanding the primary domain can change equal-peak placements
        # and introduce synchronization without buying any reuse avoidance.
        reuse_budget = bytes_budget if nonempty_pairs else 0

        seed, seed_peak = first_fit_placement(nonempty, memory_space)
        capacity = usable_capacity(memory_space)
        upper_bound = min(seed_peak + reuse_budget, capacity)
        lower_bound = _aligned_live_peak_lower_bound(nonempty, memory_space.granularity)
        if lower_bound > upper_bound:
            raise IRVerificationError(f"SAT allocation for memory space {memory_space.name!r} failed: "
                                      f"live-byte lower bound {lower_bound} exceeds capacity {capacity}.")

        # Non-overlapping lifetimes can admit many equal-peak layouts with
        # different incidental reuse hazards. Preserve their established SAT
        # placement; only simultaneous roots have an invariant no-reuse graph.
        simultaneous = max(v.live_start for v in nonempty) <= min(v.live_end for v in nonempty)
        witness = ((seed if seed_peak == lower_bound else _clique_bound_witness(nonempty, memory_space, lower_bound))
                   if simultaneous else None)
        if witness is not None and not reuse_conflicts(nonempty, witness, nonempty_pairs):
            # Feasibility plus matching bounds proves every lexicographic
            # objective: minimum raw peak, zero reuse conflicts, and the same
            # minimum peak after optional budgeted reuse avoidance. This is a
            # complete solution certificate, not a time-budget fallback.
            witness.update((value.id, 0) for value in lifetimes if not value.nbytes)
            objectives = [AllocationObjective("high_water", "OPTIMAL", lower_bound, lower_bound)]
            if nonempty_pairs:
                objectives.append(AllocationObjective("reuse_conflicts", "OPTIMAL", 0, 0))
                if reuse_budget:
                    objectives.append(AllocationObjective("peak_reclaim", "OPTIMAL", lower_bound, lower_bound))
            return verify_allocation(lifetimes, memory_space, AllocationResult(
                tuple((value.id, witness[value.id]) for value in lifetimes),
                memory_space.allocation_bytes(lower_bound), "OPTIMAL", (), tuple(objectives)))

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

        overlap_hints = []

        def add_hint(offsets, peak):
            model.clear_hints()
            model.add_hint(pool_end, peak)
            for name, (quotient, start, end, alignment, size) in variables.items():
                model.add_hint(quotient, offsets[name] // alignment)
                model.add_hint(start, offsets[name])
                model.add_hint(end, offsets[name] + size)
            for left, right, overlap, left_before, right_before in overlap_hints:
                before_left = offsets[left] + variables[left][4] <= offsets[right]
                before_right = offsets[right] + variables[right][4] <= offsets[left]
                model.add_hint(left_before, int(before_left))
                model.add_hint(right_before, int(before_right))
                model.add_hint(overlap, int(not before_left and not before_right))

        if seed_peak <= capacity:
            add_hint(seed, seed_peak)
        model.minimize(pool_end)
        solver = self._solver(deadline)
        status = solver.solve(model)
        self._require_solution(status, solver, memory_space)
        high_water_status = solver.status_name(status)
        selected_peak = solver.value(pool_end)
        proven_peak = selected_peak if high_water_status == "OPTIMAL" else lower_bound
        offsets = {name: solver.value(value[1]) for name, value in variables.items()}
        objectives = [AllocationObjective("high_water", high_water_status, selected_peak, solver.best_objective_bound)]
        if reuse_budget:
            # Trade bounded extra bytes for fewer inter-owner hazards: keep the
            # peak within the budget above its minimum, minimize reuse
            # conflicts first, then reclaim the smallest peak that achieves
            # that conflict count.  With a zero budget the pinned-peak
            # behavior below is bit-identical to the original contract.
            model.add(pool_end <= selected_peak + reuse_budget)
            model.add(pool_end >= proven_peak)
        else:
            model.add(pool_end == selected_peak)

        # Unlike sum(addresses), this objective corresponds to proven
        # inter-owner hazards. All objectives share one allocation budget.
        reuse_status = "OPTIMAL"
        reclaim_objective = None
        if nonempty_pairs:
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
                overlap_hints.append((left, right, overlap, left_before, right_before))
                overlap_vars.append(overlap)
            # Spatially disjoint cliques give the solver the packing bounds
            # that its generic 2D encoding otherwise has to rediscover.
            for minimum_peak, clique_pairs in (_disjoint_cliques(nonempty, nonempty_pairs, memory_space.granularity)
                                               if reuse_budget else ()):
                if minimum_peak > proven_peak and clique_pairs:
                    savings = sum(min(align_up(variables[nonempty_pairs[index][0]][4], memory_space.granularity),
                                      align_up(variables[nonempty_pairs[index][1]][4], memory_space.granularity))
                                  * overlap_vars[index] for index in clique_pairs)
                    model.add(pool_end + savings >= minimum_peak)
            add_hint(offsets, selected_peak)
            model.minimize(sum(overlap_vars))
            bound = None
            if perf_counter() < deadline:
                solver = self._solver(deadline)
                status = solver.solve(model)
                reuse_status = solver.status_name(status)
                if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    offsets = {name: solver.value(value[1]) for name, value in variables.items()}
                    selected_peak = solver.value(pool_end)
                    bound = solver.best_objective_bound
                    if reuse_budget:
                        conflict_count = solver.value(sum(overlap_vars))
                        peak_bound = max(proven_peak, _reuse_clique_lower_bound(
                            nonempty, nonempty_pairs, memory_space.granularity, conflict_count))
                        if peak_bound > selected_peak:
                            raise IRVerificationError("A reuse peak bound exceeds its proven feasible layout.")
                        reclaim_status = "OPTIMAL" if selected_peak == peak_bound else "NOT_RUN_BUDGET"
                        remaining = deadline - perf_counter()
                        if selected_peak != peak_bound and remaining > 0:
                            model.add(sum(overlap_vars) == conflict_count)
                            model.add(pool_end >= peak_bound)
                            model.add(pool_end <= selected_peak)
                            model.minimize(pool_end)
                            add_hint(offsets, selected_peak)
                            solver = self._solver_for_duration(remaining)
                            peak_status = solver.solve(model)
                            reclaim_status = solver.status_name(peak_status)
                            peak_bound = max(peak_bound, solver.best_objective_bound)
                            if peak_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                                offsets = {name: solver.value(value[1]) for name, value in variables.items()}
                                selected_peak = solver.value(pool_end)
                            elif peak_status != cp_model.UNKNOWN:
                                self._require_solution(peak_status, solver, memory_space)
                        reclaim_objective = AllocationObjective("peak_reclaim", reclaim_status, selected_peak, peak_bound)
                elif status != cp_model.UNKNOWN:
                    self._require_solution(status, solver, memory_space)
                # UNKNOWN retains the already proven feasible SAT placement,
                # with incomplete optimization explicitly recorded below.
            else:
                reuse_status = "NOT_RUN_BUDGET"
            conflicts = reuse_conflicts(nonempty, offsets, nonempty_pairs)
            objectives.append(AllocationObjective("reuse_conflicts", reuse_status, len(conflicts), bound))
            if reclaim_objective is not None:
                objectives.append(reclaim_objective)
        offsets.update((value.id, 0) for value in lifetimes if not value.nbytes)
        combined = ("OPTIMAL" if high_water_status == reuse_status == "OPTIMAL"
                    and (reclaim_objective is None or reclaim_objective.status == "OPTIMAL") else
                    f"high-water:{high_water_status};reuse:{reuse_status}")
        if reclaim_objective is not None and reclaim_objective.status != "OPTIMAL":
            combined += f";peak-reclaim:{reclaim_objective.status}"
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
        return self._solver_for_duration(deadline - perf_counter())

    def _solver_for_duration(self, remaining):
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
