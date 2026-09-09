# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest
from ortools.sat.python import cp_model

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.auto_distributed import AutoDistributedPass, solve_search_graph
from triton.flagmega.passes.auto_distributed.search_domains import propagate_domains
from triton.flagmega.targets import NvidiaSm90Target

from .helpers import packed_matmul_module


def _sites(graph):
    return {(s.producer_id, s.producer_index, s.consumer_id, s.consumer_index, s.input_index): s
            for s in graph.reshard_sites}


def test_fixed_choices_prune_model_without_changing_search_surface(monkeypatch):
    graph = AutoDistributedPass._build_graph(packed_matmul_module(), NvidiaSm90Target())
    surface = tuple(tuple(c.id for c in b.candidates) for b in graph.buckets)
    sizes = []
    original = cp_model.CpSolver.solve

    def capture(self, model, *args, **kwargs):
        sizes.append(len(model.proto.variables))
        return original(self, model, *args, **kwargs)

    monkeypatch.setattr(cp_model.CpSolver, "solve", capture)
    proposal = solve_search_graph(graph)
    fixed = {name: c.id for name, c in proposal.selected.items()}
    applied = solve_search_graph(graph, fixed_selections=fixed)
    assert applied.selected == proposal.selected
    assert applied.objective == proposal.objective
    assert sizes[1] < sizes[0]
    assert tuple(tuple(c.id for c in b.candidates) for b in graph.buckets) == surface


def test_propagation_matches_unpruned_cp_sat_for_every_editable_candidate(monkeypatch):
    from triton.flagmega.passes.auto_distributed import search_domains
    graph = AutoDistributedPass._build_graph(packed_matmul_module(), NvidiaSm90Target())

    def unpruned(graph, fixed, sites):
        return {
            b.node_id: tuple(i
                             for i, c in enumerate(b.candidates)
                             if b.node_id not in fixed or c.id == fixed[b.node_id])
            for b in graph.buckets
        }

    for candidate in graph.bucket_map["output"].candidates:
        fixed = {"output": candidate.id}
        with monkeypatch.context() as context:
            context.setattr(search_domains, "propagate_domains", unpruned)
            expected = solve_search_graph(graph, fixed_selections=fixed)
        actual = solve_search_graph(graph, fixed_selections=fixed)
        assert actual.objective == expected.objective
        assert actual.selected["output"].id == candidate.id


def test_impossible_output_or_fixed_choice_fails_before_sat():
    graph = AutoDistributedPass._build_graph(packed_matmul_module(), NvidiaSm90Target())
    with pytest.raises(IRVerificationError, match="unknown nodes"):
        propagate_domains(graph, {"missing": "x"}, _sites(graph))
    with pytest.raises(IRVerificationError, match="not legal"):
        propagate_domains(graph, {"output": "missing"}, _sites(graph))
    # Logical output candidates all need an explicit distributed function ABI edge.
    incompatible = next(c for c in graph.bucket_map["output"].candidates
                        if c.return_type != graph.bucket_map["output"].candidates[0].return_type)
    no_abi = replace(graph, reshard_sites=tuple(s for s in graph.reshard_sites if s.consumer_index is not None))
    with pytest.raises(IRVerificationError, match="INFEASIBLE"):
        propagate_domains(no_abi, {"output": incompatible.id}, _sites(no_abi))
