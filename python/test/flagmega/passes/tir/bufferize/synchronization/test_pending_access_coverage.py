# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.tir.bufferize.barrier_coverage import BarrierCoverage


def test_block_coverage_does_not_cover_chip_or_axis_group_hazards():
    placement = fm.Placement((2, 4), "yx", "bb")
    coverage = BarrierCoverage()
    assert not coverage.covers(("block", (), None))
    coverage.apply(("block", (), None), None)
    assert coverage.covers(("block", (), None))
    assert not coverage.covers(("grid", (0,), placement))
    assert not coverage.covers(("grid", (), placement))


def test_axis_group_coverage_accumulates_on_the_same_mesh_like_nncase():
    placement = fm.Placement((2, 4), "yx", "bb")
    coverage = BarrierCoverage()
    coverage.apply(("grid", (0,), placement), None)
    assert coverage.covers(("grid", (0,), placement))
    assert not coverage.covers(("grid", (1,), placement))
    assert not coverage.covers(("grid", (), placement))
    coverage.apply(("grid", (1,), placement), None)
    assert coverage.full_chip_synchronized
    assert coverage.covers(("grid", (), placement))


def test_axis_groups_from_different_placements_do_not_compose():
    first = fm.Placement((2, 4), "yx", "bb")
    second = fm.Placement((4, 2), "yx", "bb")
    coverage = BarrierCoverage()
    coverage.apply(("grid", (0,), first), None)
    coverage.apply(("grid", (1,), second), None)
    assert not coverage.full_chip_synchronized
    assert not coverage.covers(("grid", (0,), second))
    assert not coverage.covers(("grid", (1,), first))


def test_axis_group_barrier_cannot_cover_an_access_on_another_mesh():
    first = fm.Placement((2, 4), "yx", "bb")
    second = fm.Placement((4, 2), "yx", "bb")
    value_type = fm.DistributedType(fm.tensor_type("float32", (1,)), (fm.SBP.broadcast(),), second)
    coverage = BarrierCoverage()
    coverage.apply(("grid", (0,), first), value_type)
    assert coverage.block_synchronized
    assert not coverage.covers(("grid", (0,), first))
