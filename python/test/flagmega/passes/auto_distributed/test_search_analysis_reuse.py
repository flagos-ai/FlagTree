# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from collections import Counter
from dataclasses import replace

from triton.flagmega.passes.auto_distributed import AutoDistributedPass
from triton.flagmega.passes.auto_distributed import search
from triton.flagmega.targets import NvidiaSm90Target

from .helpers import packed_matmul_module


def test_one_search_computes_each_exact_reshard_problem_once(monkeypatch):
    counts = Counter()
    original = search._reshard_plans

    def counted(source, target, policy, source_kind, usage):
        counts[source, target, source_kind, usage] += 1
        return original(source, target, policy, source_kind, usage)

    monkeypatch.setattr(search, "_reshard_plans", counted)
    AutoDistributedPass._build_graph(packed_matmul_module(), NvidiaSm90Target())
    assert counts
    assert max(counts.values()) == 1


def test_pick_graph_contains_only_selected_candidates_and_plans():
    graph = AutoDistributedPass._build_graph(packed_matmul_module(), NvidiaSm90Target())
    result = search.solve_search_graph(graph)
    full = search.graph_dot(graph)
    picked = search.graph_dot(graph, result.selected, result.selected_reshards)
    for bucket in graph.buckets:
        for candidate in bucket.candidates:
            assert candidate.id in full
            assert (candidate.id in picked) == (candidate == result.selected[bucket.node_id])
    assert picked.count("shape=diamond") == len(result.selected_reshards)
    assert len(picked) < len(full)


def test_cost_queries_are_shared_by_solve_and_dump_but_not_replaced_graph(monkeypatch):
    from triton.flagmega.passes.auto_distributed.reshard_cost import DistributedReshardCostModel
    calls = []
    original = DistributedReshardCostModel.realization_cost

    def counted(self, context, realization):
        calls.append((context, realization))
        return original(self, context, realization)

    monkeypatch.setattr(DistributedReshardCostModel, "realization_cost", counted)
    graph = AutoDistributedPass._build_graph(packed_matmul_module(), NvidiaSm90Target())
    result = search.solve_search_graph(graph)
    search.graph_dot(graph)
    count = len(calls)
    search.graph_dot(graph, result.selected, result.selected_reshards)
    search.graph_dot(graph)
    assert count > 0 and len(calls) == count
    changed = replace(graph, reshard_cost_model=DistributedReshardCostModel(grid_synchronization_cost=7000))
    assert not changed._realized_costs
    search.graph_dot(changed)
    assert len(calls) > count
