# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm


def test_target_leaf_policies_deduplicate_and_filter_cross_axis_ownership():
    tensor = fm.tensor_type("float32", (8, 16))
    placement = fm.Placement((2, 4), "yx", "bb")
    queries = []

    def splits(value, axis, hierarchy_axes):
        queries.append((value, axis, hierarchy_axes))
        policy = fm.SBP.split_block_cyclic(hierarchy_axes, 1)
        return (policy, policy)

    policies = fm.leaf_candidate_policies(tensor, placement, split_candidates=splits)

    assert len(queries) == 6
    assert len(policies) == len(set(policies))
    assert len(policies) == len(fm.leaf_candidate_policies(tensor, placement))
    for candidate in policies:
        owners = [owner for sbp in candidate if isinstance(sbp, fm.SBPSplit)
                  for owner in sbp.hierarchy_axes]
        assert len(owners) == len(set(owners))
        assert fm.is_distributable(tensor, candidate, placement)


def test_target_can_decline_splits_without_implicit_contiguous_fallback():
    tensor = fm.tensor_type("bfloat16", (1, 128))
    placement = fm.Placement((2, 4), "yx", "bb")

    assert fm.leaf_candidate_policies(tensor, placement, split_candidates=lambda *args: ()) == (
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
    )
