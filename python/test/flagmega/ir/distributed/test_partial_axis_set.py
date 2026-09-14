# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm


def test_partial_reduction_axes_are_a_canonical_set_not_split_stage_order():
    split = fm.SBP.split(fm.SplitStage.block_cyclic((1,), 32), fm.SplitStage.block_cyclic((0,), 4))
    assert split.hierarchy_axes == (1, 0)
    partial = fm.SBP.partial(split.hierarchy_axes)
    assert partial.axes == (0, 1)
    assert partial == fm.SBP.partial((0, 1))
    assert partial.to_data()["axes"] == [0, 1]
    assert split.hierarchy_axes == (1, 0)
