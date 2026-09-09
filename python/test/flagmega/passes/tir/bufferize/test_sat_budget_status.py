# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
from ortools.sat.python import cp_model

from triton.flagmega.ir.bufferization import AllocationStrategy, MemorySpace
from triton.flagmega.passes.tir.bufferize import BufferLifetime, SATBufferAllocator


@pytest.mark.parametrize("first_status", [cp_model.FEASIBLE, cp_model.OPTIMAL])
def test_incomplete_secondary_optimization_retains_verified_sat_incumbent(monkeypatch, first_status):
    original = cp_model.CpSolver.solve
    calls = []

    def solve(self, model, *args, **kwargs):
        calls.append(self.parameters.max_time_in_seconds)
        if len(calls) == 2:
            return cp_model.UNKNOWN
        assert original(self, model, *args, **kwargs) == cp_model.OPTIMAL
        return first_status

    monkeypatch.setattr(cp_model.CpSolver, "solve", solve)
    result = SATBufferAllocator().allocate((
        BufferLifetime("a", 64, 64, 0, 0),
        BufferLifetime("b", 64, 64, 1, 1),
    ), MemorySpace("workspace", "device", 64, 4096, AllocationStrategy.REUSE), avoid_reuse=(("a", "b"), ))
    assert result.pool_bytes == 64
    assert result.reuse_conflicts == (("a", "b"), )
    assert result.status != "OPTIMAL"
    assert result.objectives[0].status == ("OPTIMAL" if first_status == cp_model.OPTIMAL else "FEASIBLE")
    assert result.objectives[1].status == "UNKNOWN"
    assert result.objectives[1].best_bound is None
    assert len(calls) == 2 and calls[1] < calls[0]
