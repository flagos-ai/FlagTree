# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Reuse and peak reclamation must retain a feasible, honestly sized plan."""

from ortools.sat.python import cp_model

from triton.flagmega.ir.bufferization import (
    AllocationStrategy, MemoryAllocationScope, MemorySharingScope, MemorySpace,
)
from triton.flagmega.passes.tir.bufferize.allocation import BufferLifetime, verify_allocation
from triton.flagmega.passes.tir.bufferize.sat_allocator import SATBufferAllocator


def _exercise_reclaim_handoff(monkeypatch):
    from triton.flagmega.passes.tir.bufferize import sat_allocator
    # Isolate handoff correctness when no extra proof closes the peak gap.
    monkeypatch.setattr(sat_allocator, "_disjoint_clique_lower_bound", lambda *args: 0, raising=False)


def _space():
    return MemorySpace("workspace", "device", 64, 4096, AllocationStrategy.SAT,
                       allocation_scope=MemoryAllocationScope.FUNCTION,
                       sharing_scope=MemorySharingScope.CHIP)


def test_unknown_reclaim_keeps_the_enlarged_feasible_pool(monkeypatch):
    _exercise_reclaim_handoff(monkeypatch)
    real_solve = cp_model.CpSolver.solve
    phases = []

    def solve(solver, model, *args, **kwargs):
        phases.append(model)
        if len(phases) == 3:
            # Real CP-SAT deadline exhaustion, after two real optimal solves.
            solver.parameters.max_time_in_seconds = 1e-10
        status = real_solve(solver, model, *args, **kwargs)
        if len(phases) == 3:
            assert status == cp_model.UNKNOWN
        return status

    monkeypatch.setattr(cp_model.CpSolver, "solve", solve)
    lifetimes = (BufferLifetime("old", 64, 64, 0, 0), BufferLifetime("new", 64, 64, 1, 1))
    result = SATBufferAllocator().allocate(lifetimes, _space(), avoid_reuse=(("old", "new"),), bytes_budget=64)
    assert len(phases) == 3
    assert result.pool_bytes == 128 and not result.reuse_conflicts
    assert result.offset_map["old"] != result.offset_map["new"]
    assert verify_allocation(lifetimes, _space(), result) is result
    assert result.status != "OPTIMAL"
    assert {value.name: value.status for value in result.objectives}["peak_reclaim"] == "UNKNOWN"


def test_minimum_peak_reuse_solution_needs_no_redundant_search(monkeypatch):
    real_solve = cp_model.CpSolver.solve
    phases = []

    def solve(solver, model, *args, **kwargs):
        phases.append(model)
        assert len(phases) <= 2, "An optimal peak plus a feasible layout at that peak already proves reclamation."
        return real_solve(solver, model, *args, **kwargs)

    monkeypatch.setattr(cp_model.CpSolver, "solve", solve)
    lifetimes = (BufferLifetime("anchor", 128, 64, 0, 0),
                 BufferLifetime("old", 64, 64, 1, 1), BufferLifetime("new", 64, 64, 2, 2))
    result = SATBufferAllocator().allocate(lifetimes, _space(), avoid_reuse=(("old", "new"),), bytes_budget=1)
    assert result.pool_bytes == 128 and not result.reuse_conflicts
    assert result.status == "OPTIMAL"
    assert {value.name: value.status for value in result.objectives}["peak_reclaim"] == "OPTIMAL"


def test_reclaim_starts_with_the_current_feasible_layout(monkeypatch):
    _exercise_reclaim_handoff(monkeypatch)
    real_solve = cp_model.CpSolver.solve
    phases = []

    def solve(solver, model, *args, **kwargs):
        phases.append(model)
        if len(phases) == 3:
            # All hinted values must satisfy the newly fixed conflict count.
            solver.parameters.fix_variables_to_their_hinted_value = True
        return real_solve(solver, model, *args, **kwargs)

    monkeypatch.setattr(cp_model.CpSolver, "solve", solve)
    lifetimes = (BufferLifetime("old", 64, 64, 0, 0), BufferLifetime("new", 64, 64, 1, 1))
    result = SATBufferAllocator().allocate(lifetimes, _space(), avoid_reuse=(("old", "new"),), bytes_budget=64)
    assert len(phases) == 3 and result.pool_bytes == 128 and not result.reuse_conflicts


def test_exhausted_reclaim_budget_keeps_the_proven_feasible_layout(monkeypatch):
    _exercise_reclaim_handoff(monkeypatch)
    from triton.flagmega.passes.tir.bufferize import sat_allocator
    real_solve = cp_model.CpSolver.solve
    real_clock = sat_allocator.perf_counter
    solves = []

    def solve(solver, model, *args, **kwargs):
        status = real_solve(solver, model, *args, **kwargs)
        solves.append(status)
        return status

    def clock():
        return real_clock() + (100 if len(solves) >= 2 else 0)

    monkeypatch.setattr(cp_model.CpSolver, "solve", solve)
    monkeypatch.setattr(sat_allocator, "perf_counter", clock)
    lifetimes = (BufferLifetime("old", 64, 64, 0, 0), BufferLifetime("new", 64, 64, 1, 1))
    result = SATBufferAllocator().allocate(lifetimes, _space(), avoid_reuse=(("old", "new"),), bytes_budget=64)
    assert len(solves) == 2 and result.pool_bytes == 128 and not result.reuse_conflicts
    assert {value.name: value.status for value in result.objectives}["peak_reclaim"] == "NOT_RUN_BUDGET"
    assert result.status != "OPTIMAL"


def test_zero_conflict_clique_proves_reclaimed_peak_without_another_solve(monkeypatch):
    real_solve = cp_model.CpSolver.solve
    phases = []

    def solve(solver, model, *args, **kwargs):
        phases.append(model)
        assert len(phases) <= 2, "Two disjoint 64-byte roots prove a 128-byte peak."
        return real_solve(solver, model, *args, **kwargs)

    monkeypatch.setattr(cp_model.CpSolver, "solve", solve)
    lifetimes = (BufferLifetime("old", 64, 64, 0, 0), BufferLifetime("new", 64, 64, 1, 1))
    result = SATBufferAllocator().allocate(lifetimes, _space(), avoid_reuse=(("old", "new"),), bytes_budget=64)
    assert result.pool_bytes == 128 and not result.reuse_conflicts and result.status == "OPTIMAL"


def test_disjoint_clique_bound_is_valid_against_exhaustive_graph_oracle():
    from itertools import combinations
    from random import Random
    from triton.flagmega.passes.tir.bufferize.sat_allocator import _disjoint_clique_lower_bound

    random = Random(19)
    for case in range(40):
        lifetimes = tuple(BufferLifetime(str(i), random.choice((1, 17, 32, 47)), 16,
                                         i % 3, i % 3 + random.randrange(2)) for i in range(7))
        pairs = tuple((a.id,b.id) for a,b in combinations(lifetimes,2) if random.randrange(3)==0)
        edges = {frozenset(pair)for pair in pairs}
        oracle = 0
        for mask in range(1, 1 << len(lifetimes)):
            clique = tuple(value for i,value in enumerate(lifetimes) if mask & (1 << i))
            if not all((a.live_start <= b.live_end and b.live_start <= a.live_end)
                       or frozenset((a.id,b.id)) in edges for a,b in combinations(clique,2)):
                continue
            padded = tuple((value.nbytes+15)//16*16 for value in clique)
            oracle = max(oracle, sum(padded)-max(size-value.nbytes for size,value in zip(padded,clique)))
        bound = _disjoint_clique_lower_bound(lifetimes,pairs,16)
        assert 0 <= bound <= oracle, (case,bound,oracle)


def test_positive_budget_objectives_match_exhaustive_address_placement():
    from itertools import combinations, product
    from random import Random
    random = Random(61)
    for case in range(8):
        lifetimes = tuple(BufferLifetime(str(i), random.choice((64,128)), 64,
                                         i % 3, i % 3 + random.randrange(2)) for i in range(4))
        pairs = tuple((a.id,b.id) for a,b in combinations(lifetimes,2) if random.randrange(3))
        candidates = []
        for offsets in product(*(range(0,513-v.nbytes,64)for v in lifetimes)):
            def overlaps(i,j):
                return offsets[i] < offsets[j]+lifetimes[j].nbytes and offsets[j] < offsets[i]+lifetimes[i].nbytes
            if any(overlaps(i,j) and lifetimes[i].live_start <= lifetimes[j].live_end
                   and lifetimes[j].live_start <= lifetimes[i].live_end for i,j in combinations(range(4),2)):
                continue
            peak = max(offset+v.nbytes for offset,v in zip(offsets,lifetimes))
            conflicts = sum(overlaps(int(a),int(b))for a,b in pairs)
            candidates.append((conflicts,peak))
        minimum = min(peak for _,peak in candidates)
        for budget in (64,128):
            expected = min((conflicts,peak)for conflicts,peak in candidates if peak <= minimum+budget)
            result = SATBufferAllocator().allocate(lifetimes,_space(),avoid_reuse=pairs,bytes_budget=budget)
            assert (len(result.reuse_conflicts),result.pool_bytes) == expected, (case,budget,expected,result)
            assert result.status == "OPTIMAL"
