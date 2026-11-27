# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Budget exhaustion must not be reported as a distribution failure.

The 45-layer GLM-5.3-Flash model builds a 604k-boolean CP-SAT model whose
objective spans 46 bits, so the optimiser cannot close it: with the objective
present it reports ``FEASIBLE`` after 1416 deterministic units, while pure
feasibility on the same model is ``OPTIMAL`` after 0.9.  Under a wall-clock
deadline the pass therefore succeeded on an idle host and raised ``UNKNOWN``
on a loaded one for the same input.  ``UNKNOWN`` is not a statement about the
module -- infeasible candidate relations are reported as ``INFEASIBLE`` by
``propagate_domains`` -- so it must never surface as a compile error.
"""

import pytest
from ortools.sat.python import cp_model

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.auto_distributed import (
    AutoDistributedPass,
    solve_search_graph,
)
from triton.flagmega.passes.auto_distributed import search as search_module
from triton.flagmega.targets import NvidiaSm90Target

from .helpers import packed_matmul_module


def _starve_optimizer(monkeypatch):
    """Let feasibility succeed while the objective search runs out of budget.

    This is what the real model does at scale: the first feasible assignment
    is cheap, and only the objective-driven search exhausts its budget.
    """

    original = cp_model.CpSolver.Solve

    def solve(self, model, *args, **kwargs):
        if not self.parameters.stop_after_first_solution:
            self.parameters.max_deterministic_time = 1e-9
        return original(self, model, *args, **kwargs)

    monkeypatch.setattr(cp_model.CpSolver, "Solve", solve)


def test_exhausted_objective_budget_still_yields_a_distribution(monkeypatch):
    graph = AutoDistributedPass._build_graph(
        packed_matmul_module(), NvidiaSm90Target())
    reference = solve_search_graph(graph)

    _starve_optimizer(monkeypatch)
    result = solve_search_graph(graph)

    # A plan, not an exception: every bucket is assigned and every selection
    # is one of the candidates the graph actually offers.
    assert set(result.selected) == {bucket.node_id for bucket in graph.buckets}
    for bucket in graph.buckets:
        assert result.selected[bucket.node_id] in bucket.candidates
    assert result.objective >= reference.objective


def test_infeasible_model_is_still_reported(monkeypatch):
    """Only genuine infeasibility remains an extraction failure."""

    graph = AutoDistributedPass._build_graph(
        packed_matmul_module(), NvidiaSm90Target())
    original = solve_search_graph.__globals__["_solve_distribution_model"]

    def contradictory(cp_model_module, model, search_graph):
        variable = model.NewBoolVar("infeasible_probe")
        model.Add(variable == 0)
        model.Add(variable == 1)
        return original(cp_model_module, model, search_graph)

    monkeypatch.setattr(
        search_module, "_solve_distribution_model", contradictory)
    with pytest.raises(IRVerificationError, match="INFEASIBLE"):
        solve_search_graph(graph)


def test_budget_is_deterministic_not_wall_clock(monkeypatch):
    """Machine load must not decide whether a module compiles."""

    graph = AutoDistributedPass._build_graph(
        packed_matmul_module(), NvidiaSm90Target())
    seen = []
    original = cp_model.CpSolver.Solve

    def solve(self, model, *args, **kwargs):
        seen.append((
            self.parameters.max_time_in_seconds,
            self.parameters.max_deterministic_time,
            self.parameters.interleave_search,
        ))
        return original(self, model, *args, **kwargs)

    monkeypatch.setattr(cp_model.CpSolver, "Solve", solve)
    solve_search_graph(graph)

    assert seen
    for wall_clock, deterministic, interleave in seen:
        # OR-Tools defaults max_time_in_seconds to +inf when it is unset.
        assert wall_clock == float("inf")
        assert 0 < deterministic < float("inf")
        assert interleave
