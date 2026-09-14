# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.codegen.triton.candidates.core import TritonCandidateContext
from triton.flagmega.codegen.triton.candidates.sparse_experts import SparseExpertsCandidateProvider
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from triton.flagmega.targets import NvidiaSm90Target
from python.test.flagmega.codegen.triton.kernels.sparse_experts.helpers import stage_module


def test_pipeline_catalog_declares_shared_alignment_before_injection():
    from triton.flagmega.targets.nvidia import sm90_triton_implementation_model
    from triton.flagmega.targets.nvidia.shared_layout import verify_shared_workspaces
    model = sm90_triton_implementation_model()
    verify_shared_workspaces(model)
    NvidiaSm90Target(triton_implementation_model=model)


@pytest.mark.parametrize("definition", (SparseExpertsGateUp, SparseExpertsDown))
@pytest.mark.parametrize("dtype", ("bfloat16", "float32"))
def test_expert_tma_candidate_has_dynamic_route_dependency(definition, dtype):
    graph = stage_module(definition, dtype=dtype)
    target = NvidiaSm90Target()
    context = TritonCandidateContext(graph, target, {}, {}, {}, frozenset(), True,
                                    target.triton_implementation_model)
    proposal = SparseExpertsCandidateProvider().propose(graph.node_map["expert_stage"], context)
    pipelined = [context.implementation_model.implementation(candidate.id)
                 for candidate in proposal.candidates if "simt_tma_pipeline" in candidate.id]
    assert len(pipelined) == 1
    implementation = pipelined[0]
    contract = implementation.transfer_pipeline
    assert contract.producer_read_argument_indices == (1,)
    assert 1 not in contract.source_argument_indices
    assert implementation.contract["required_weight_dtype"] == dtype


@pytest.mark.parametrize("definition", (SparseExpertsGateUp, SparseExpertsDown))
@pytest.mark.parametrize("reason", ("nonaffine_owner", "unaligned_row"))
def test_invalid_weight_transport_is_excluded_before_selection(definition, reason):
    options = ({"distribution": "token_output"} if reason == "nonaffine_owner" else
               {"hidden": 74, "intermediate": 42})
    graph = stage_module(definition, **options)
    original_hash = graph.semantic_hash
    target = NvidiaSm90Target()
    context = TritonCandidateContext(graph, target, {}, {}, {}, frozenset(), True,
                                    target.triton_implementation_model)
    proposal = SparseExpertsCandidateProvider().propose(graph.node_map["expert_stage"], context)
    assert proposal is not None
    assert all("simt_tma_pipeline" not in candidate.id for candidate in proposal.candidates)
    assert any(candidate.id.endswith(".simt") for candidate in proposal.candidates)
    assert graph.semantic_hash == original_hash
