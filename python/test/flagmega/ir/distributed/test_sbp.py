# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError


def test_placement_normalizes_names_levels_and_physical_sizes():
    placement = fm.Placement((2, 4), "c, b", "d / b")

    assert placement.rank == 2
    assert placement.size == 8
    assert placement.normalized_hierarchy_names == "cb"
    assert placement.normalized_hierarchy_levels == "db"
    assert placement.physical_level_size("b") == 4
    assert placement.is_physical_block_axis(1)


def test_split_stages_preserve_contiguous_and_block_cyclic_semantics():
    contiguous = fm.SBP.split_contiguous((0,), granularity=16)
    block_cyclic = fm.SBP.split_block_cyclic((1,), block_size=8)

    assert contiguous.is_contiguous
    assert not block_cyclic.is_contiguous
    assert contiguous.hierarchy_axes == (0,)
    assert str(block_cyclic) == "S(BC(8)@[1])"


def test_distributed_type_rejects_reused_hierarchy_axis_and_non_divisible_extent():
    placement = fm.Placement((8,), "b", "b")
    tensor = fm.tensor_type("bfloat16", [4, 16])

    with pytest.raises(IRSchemaError, match="not distributable"):
        fm.DistributedType(
            tensor,
            (fm.SBP.split_contiguous((0,)), fm.SBP.split_contiguous((0,))),
            placement,
        )


def test_block_cyclic_split_accepts_a_non_divisible_tail_block():
    placement = fm.Placement((8, 16), "yx", "bb")
    tensor = fm.tensor_type("bfloat16", [8, 18992, 2, 64])

    distributed = fm.DistributedType(
        tensor,
        (
            fm.SBP.broadcast(),
            fm.SBP.split_block_cyclic((0, 1), block_size=8),
            fm.SBP.broadcast(),
            fm.SBP.broadcast(),
        ),
        placement,
    )

    assert distributed.axis_policies[1].hierarchy_axes == (0, 1)
    with pytest.raises(IRSchemaError, match="not distributable"):
        fm.DistributedType(
            tensor,
            (fm.SBP.split_contiguous((0,)), fm.SBP.broadcast()),
            placement,
        )


def test_leaf_candidates_match_nncase_cartesian_policy_contract():
    placement = fm.Placement((2, 4), "db", "db")
    tensor = fm.tensor_type("float32", [8, 16])
    candidates = fm.leaf_candidate_policies(tensor, placement)

    assert (fm.SBP.broadcast(), fm.SBP.broadcast()) in candidates
    assert any(
        isinstance(candidate[0], fm.SBPSplit)
        and candidate[0].hierarchy_axes == (0, 1)
        and isinstance(candidate[1], fm.SBPBroadCast)
        for candidate in candidates
    )
    assert all(fm.is_distributable(tensor, candidate, placement) for candidate in candidates)
