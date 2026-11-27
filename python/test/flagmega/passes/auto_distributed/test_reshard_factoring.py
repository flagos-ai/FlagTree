# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Implementation variants must not multiply identical layout relations."""

from dataclasses import replace
from itertools import product

import pytest
from triton.flagmega.passes.auto_distributed import AutoDistributedPass, search, solve_search_graph
from triton.flagmega.passes.auto_distributed.reshard_cost import DistributedReshardCostModel
from triton.flagmega.targets import NvidiaSm90Target

from .helpers import packed_matmul_module


def _variants(graph, count):
    buckets = tuple(
        replace(
            bucket,
            candidates=tuple(
                replace(
                    candidate,
                    id=f"{candidate.id}:variant{variant}",
                    operation_cost=candidate.operation_cost + count - 1 - variant,
                )
                for candidate in bucket.candidates
                for variant in range(count)
            ),
        )
        for bucket in graph.buckets
    )
    sites = tuple(
        replace(
            site,
            producer_index=site.producer_index * count + producer,
            consumer_index=(None if consumer is None else site.consumer_index * count + consumer),
        )
        for site in graph.reshard_sites
        for producer, consumer in product(range(count), (None,) if site.consumer_index is None else range(count))
    )
    return replace(graph, buckets=buckets, reshard_sites=sites)


@pytest.mark.parametrize("fixed", [False, True])
def test_same_layout_implementation_variants_share_reshard_solver_variables(monkeypatch, fixed):
    graph = AutoDistributedPass._build_graph(packed_matmul_module(), NvidiaSm90Target())
    reference = solve_search_graph(graph)
    expanded = _variants(graph, 6)
    solve = search._solve_distribution_model

    def measure(cp_model, model, graph):
        # Candidate choices remain distinct. Adapters grow with layouts,
        # rather than the Cartesian product of implementation variants.
        candidate_count = sum(len(bucket.candidates) for bucket in graph.buckets)
        assert len(model.Proto().variables) <= candidate_count + 8 * len(reference.graph.reshard_sites)
        return solve(cp_model, model, graph)

    monkeypatch.setattr(search, "_solve_distribution_model", measure)
    constraints = {"output": reference.selected["output"].id + ":variant5"} if fixed else None
    result = solve_search_graph(expanded, fixed_selections=constraints)
    assert result.objective == reference.objective
    assert all(candidate.id.endswith(":variant5") for candidate in result.selected.values())
    assert result.selected_reshards == reference.selected_reshards


def test_zero_cost_independent_adapter_paths_need_no_solver_decisions(monkeypatch):
    class OfflineCost(DistributedReshardCostModel):
        def realization_cost(self, context, realization):
            return 0

        def shared_publication_cost(self, context, realization):
            return 0

    graph = AutoDistributedPass._build_graph(packed_matmul_module(), NvidiaSm90Target())
    graph = replace(graph, reshard_cost_model=OfflineCost())
    solve = search._solve_distribution_model

    def measure(cp_model, model, graph):
        assert len(model.Proto().variables) == sum(len(bucket.candidates) for bucket in graph.buckets)
        return solve(cp_model, model, graph)

    monkeypatch.setattr(search, "_solve_distribution_model", measure)
    result = solve_search_graph(graph)
    assert result.objective == sum(candidate.operation_cost for candidate in result.selected.values())
    assert result.selected_reshards
