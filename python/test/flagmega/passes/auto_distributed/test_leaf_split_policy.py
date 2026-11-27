# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Leaf candidates must use the target's split contract, as in nncase."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import (
    DistributedCandidateContext,
    NormApplyCandidateProvider,
    NormStatsCandidateProvider,
)
from .test_norm_candidate_providers import _Module


class _CyclicPolicy:
    identity = "test-cyclic/v1"

    def __init__(self):
        self.queries = []

    def get_candidates(self, context):
        self.queries.append(context)
        return (fm.SBP.split_block_cyclic(context.hierarchy_axes, 1),)


@pytest.mark.parametrize("node_id,provider", (
    ("stats", NormStatsCandidateProvider),
    ("output", NormApplyCandidateProvider),
))
def test_norm_leaf_enumeration_uses_target_policy_without_producer_hints(node_id, provider):
    module = _Module().build()
    node = module.node_map[node_id]
    placement = fm.Placement((2, 4), "yx", "bb")
    policy = _CyclicPolicy()
    context = DistributedCandidateContext(
        module, node, placement, ((),) * len(node.inputs),
        split_candidate_provider=policy,
    )

    candidates = provider().get_candidates(context)

    expected = fm.DistributedType(
        module.node_map[node.inputs[0]].type,
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0, 1), 1)), placement,
    )
    assert any(candidate.input_types[0] == expected for candidate in candidates)
    assert {query.hierarchy_axes for query in policy.queries} == {(0,), (1,), (0, 1)}
    assert all(query.tensor_axis == 1 and query.purpose == "generic" for query in policy.queries)
    assert all(
        not isinstance(sbp, fm.SBPSplit) or isinstance(sbp.stages[0].distribution, fm.BlockCyclicSplit)
        for candidate in candidates for sbp in candidate.input_types[0].axis_policies
    )


def test_context_leaf_types_match_portable_default_and_change_with_target():
    module = _Module().build()
    node = module.node_map["stats"]
    tensor = module.node_map[node.inputs[0]].type
    placement = fm.Placement((2, 4), "yx", "bb")
    context = DistributedCandidateContext(module, node, placement, ((),))

    portable = context.leaf_candidate_types(tensor)

    assert portable == tuple(fm.DistributedType(tensor, policies, placement)
                             for policies in fm.leaf_candidate_policies(tensor, placement))
    cyclic = replace(context, split_candidate_provider=_CyclicPolicy()).leaf_candidate_types(tensor)
    assert cyclic != portable
    assert len(cyclic) == len(portable) == 4
