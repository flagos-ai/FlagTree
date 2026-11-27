# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A feasible placement attaining every objective bound needs no SAT search."""
from itertools import product

import pytest
from ortools.sat.python import cp_model
from triton.flagmega.ir.bufferization import AllocationStrategy, MemorySpace
from triton.flagmega.passes.tir.bufferize import BufferLifetime, SATBufferAllocator


@pytest.mark.parametrize("budget", [0, 8])
def test_simultaneous_aligned_roots_use_a_proven_witness_without_search(monkeypatch, budget):
    calls = []
    original = cp_model.CpSolver.solve
    def solve(self, model, *args, **kwargs):
        calls.append(True)
        return original(self, model, *args, **kwargs)
    monkeypatch.setattr(cp_model.CpSolver, "solve", solve)
    lifetimes = (BufferLifetime("tiny", 1, 4, 0, 4), BufferLifetime("wide", 5, 4, 1, 4),
                 BufferLifetime("last", 4, 4, 2, 4))
    space = MemorySpace("workspace", "device", 4, 20, AllocationStrategy.REUSE)
    feasible = []
    for offsets in product(range(0, 20, 4), repeat=3):
        if all(offsets[a] + lifetimes[a].nbytes <= offsets[b] or offsets[b] + lifetimes[b].nbytes <= offsets[a]
               for a in range(3) for b in range(a)):
            feasible.append(max(offset + v.nbytes for offset, v in zip(offsets, lifetimes)))
    result = SATBufferAllocator().allocate(lifetimes, space, avoid_reuse=(("tiny", "last"),), bytes_budget=budget)
    assert result.objectives[0].value == min(feasible) == 13
    assert result.status == "OPTIMAL" and result.reuse_conflicts == ()
    assert all(o.value == o.best_bound for o in result.objectives)
    assert not calls


def test_large_alignment_holes_still_require_search(monkeypatch):
    calls = []
    original = cp_model.CpSolver.solve
    def solve(self, model, *args, **kwargs):
        calls.append(True)
        return original(self, model, *args, **kwargs)
    monkeypatch.setattr(cp_model.CpSolver, "solve", solve)
    lifetimes = tuple(BufferLifetime(str(i), size, 16, 0, 1) for i, size in enumerate((1, 16, 1)))
    result = SATBufferAllocator().allocate(lifetimes, MemorySpace("workspace", "device", 4, 64, AllocationStrategy.REUSE))
    assert calls
    assert result.objectives[0].value == 33


def test_claimed_clique_certificates_match_exhaustive_mixed_alignment_placements():
    from triton.flagmega.passes.tir.bufferize.sat_allocator import _aligned_live_peak_lower_bound, _clique_bound_witness

    space = MemorySpace("workspace", "device", 4, 32, AllocationStrategy.REUSE)
    checked = 0
    for sizes in product(range(1, 5), repeat=3):
        for alignments in product((4, 8), repeat=3):
            values = tuple(BufferLifetime(str(i), size, alignment, 0, 1)
                           for i, (size, alignment) in enumerate(zip(sizes, alignments)))
            witness = _clique_bound_witness(values, space, _aligned_live_peak_lower_bound(values, 4))
            if witness is None:
                continue
            feasible = []
            for offsets in product(*(range(0, 32, alignment) for alignment in alignments)):
                if all(offsets[a] + sizes[a] <= offsets[b] or offsets[b] + sizes[b] <= offsets[a]
                       for a in range(3) for b in range(a)):
                    feasible.append(max(offset + size for offset, size in zip(offsets, sizes)))
            assert max(witness[str(i)] + size for i, size in enumerate(sizes)) == min(feasible)
            checked += 1
    assert checked > 100
