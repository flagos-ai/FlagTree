# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.targets.pyntt import paged_attention_split_plan


def test_split_plan_uses_smallest_fixed_physical_block_axis_like_nncase():
    placements = (
        fm.Placement((8, 16), "yx", "bb"),
        fm.Placement((8, 16), "yx", "bb"),
    )

    assert paged_attention_split_plan(placements) == (0, 8)


def test_split_plan_ignores_non_block_and_inconsistent_extent_axes():
    placements = (
        fm.Placement((8, 16, 4), "yxc", "cbb"),
        fm.Placement((8, 16, 8), "yxc", "cbb"),
    )

    assert paged_attention_split_plan(placements) == (1, 16)


@pytest.mark.parametrize(
    "placements",
    (
        (),
        (fm.Placement((1, 1), "yx", "bb"),),
        (
            fm.Placement((8,), "x", "b"),
            fm.Placement((8, 16), "yx", "bb"),
        ),
    ),
)
def test_split_plan_rejects_missing_compatible_physical_block_axis(placements):
    with pytest.raises(ValueError, match="paged-attention decomposition"):
        paged_attention_split_plan(placements)
