# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Compilation work must follow useful objectives and changed constraints."""

from ortools.sat.python import cp_model

from triton.flagmega.ir.bufferization import AllocationStrategy, MemorySpace
from triton.flagmega.passes.tir.bufferize import BufferLifetime, SATBufferAllocator


def test_sat_without_preferences_solves_only_memory_high_water(monkeypatch):
    objectives = []
    original = cp_model.CpSolver.solve

    def solve(self, model, *args, **kwargs):
        objectives.append(tuple(model.proto.variables[index].name for index in model.proto.objective.vars))
        return original(self, model, *args, **kwargs)

    monkeypatch.setattr(cp_model.CpSolver, "solve", solve)
    result = SATBufferAllocator().allocate((
        BufferLifetime("long", 192, 64, 0, 4),
        BufferLifetime("early", 320, 64, 0, 1),
        BufferLifetime("late", 320, 64, 2, 4),
    ), MemorySpace("workspace", "device", 64, 4096, AllocationStrategy.SAT))
    assert result.pool_bytes == 512
    assert objectives == [("pool_end", )]


def test_sat_reuse_phase_receives_complete_feasible_hint(monkeypatch):
    hints = []
    limits = []
    original = cp_model.CpSolver.solve

    def solve(self, model, *args, **kwargs):
        hints.append(len(model.proto.solution_hint.vars))
        limits.append(self.parameters.max_time_in_seconds)
        return original(self, model, *args, **kwargs)

    monkeypatch.setattr(cp_model.CpSolver, "solve", solve)
    result = SATBufferAllocator().allocate((
        BufferLifetime("anchor", 256, 64, 0, 0),
        BufferLifetime("early", 64, 64, 1, 1),
        BufferLifetime("late", 128, 64, 2, 2),
    ), MemorySpace("workspace", "device", 64, 4096, AllocationStrategy.SAT), avoid_reuse=(("early", "late"), ))
    assert result.pool_bytes == 256
    assert result.reuse_conflicts == ()
    assert len(hints) == 2
    assert all(hint > 0 for hint in hints)
    assert limits[1] < limits[0]
