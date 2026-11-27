# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.passes.auto_distributed import (
    DistributedCandidate,
    DistributedCandidateProviderRegistry,
    DistributedMaterializer,
    build_search_graph,
    solve_search_graph,
)
from triton.flagmega.targets import NvidiaSm90Target

from .helpers import packed_matmul_module


def test_candidate_objective_rejects_values_the_solver_used_to_clamp():
    value_type = packed_matmul_module().node_map["output"].type
    with pytest.raises(ValueError, match=r"\[0, 2_000_000_000\]"):
        DistributedCandidate("negative", value_type, (), -1, "invalid-negative")
    with pytest.raises(ValueError, match=r"\[0, 2_000_000_000\]"):
        DistributedCandidate("overflow", value_type, (), 2_000_000_001, "invalid-overflow")


def test_candidate_objective_has_model_and_evidence():
    value_type = packed_matmul_module().node_map["output"].type
    candidate = DistributedCandidate("candidate", value_type, (), 17, "unit-test")

    assert candidate.objective_kind == "heuristic"
    assert candidate.objective_model == "flagmega.distributed-work/v1"
    assert candidate.objective_evidence == ("candidate-reason:unit-test",)


def test_materialized_selection_preserves_objective_provenance():
    module = packed_matmul_module()
    target = NvidiaSm90Target()
    registry = DistributedCandidateProviderRegistry()
    target.register_auto_distributed_candidate_providers(registry)
    graph = build_search_graph(
        module,
        target.distributed_placements(module)[0],
        registry,
        target.distributed_reshard_realization_policy(),
    )
    result = solve_search_graph(graph)
    materialized = DistributedMaterializer(result, policy=target.policy_version).run()
    point = next(value for value in materialized.selection_points if value.kind == "distribution")
    selected = materialized.selection_map[point.id]
    candidate = next(value for value in point.candidates if value.id == selected.candidate_id)

    assert candidate.facts["objective"]["kind"] == "heuristic"
    assert candidate.facts["objective"]["model"] == "flagmega.distributed-work/v1"
    assert any(value.startswith("candidate-reason:") for value in selected.evidence)
