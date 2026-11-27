# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Which boxing transitions have to read a peer chip's memory instance."""

from triton.flagmega import ir as fm


PLACEMENT = fm.Placement((4, 2, 4), "cyx", "cbb")
TENSOR = fm.tensor_type("float32", (8, 64))


def _distributed(*policies, partial=None, exclusive=None, placement=PLACEMENT, tensor=TENSOR):
    return fm.DistributedType(tensor, policies, placement, partial, exclusive)


def test_chip_split_source_to_plain_tensor_crosses():
    source = _distributed(fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 16))

    assert fm.boxing_crossing_chip_axes(source, TENSOR) == (0,)
    assert fm.boxing_reads_peer_chips(source, TENSOR)


def test_split_over_chip_and_block_axes_crosses_only_the_chip_axis():
    source = _distributed(fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1), 8))

    assert fm.boxing_crossing_chip_axes(source, TENSOR) == (0,)


def test_block_only_split_stays_chip_local():
    source = _distributed(fm.SBP.broadcast(), fm.SBP.split_contiguous((1, 2), 8))

    assert fm.boxing_crossing_chip_axes(source, TENSOR) == ()
    assert not fm.boxing_reads_peer_chips(source, TENSOR)


def test_identical_chip_region_stays_chip_local():
    source = _distributed(fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1), 8))
    target = _distributed(fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 2), 4))

    # Both types give chip c the same 16-element region; only the block-level
    # refinement inside that region differs.
    assert fm.boxing_crossing_chip_axes(source, target) == ()


def test_block_refinement_of_the_same_chip_region_stays_chip_local():
    source = _distributed(fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 16))
    target = _distributed(fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1, 2), 2))

    assert fm.boxing_crossing_chip_axes(source, target) == ()


def test_chip_region_coarsening_crosses():
    source = _distributed(fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1), 8))
    target = _distributed(fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 32))

    assert fm.boxing_crossing_chip_axes(source, target) == (0,)


def test_plain_tensor_source_never_crosses():
    target = _distributed(fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 16))

    assert fm.boxing_crossing_chip_axes(TENSOR, target) == ()


def test_partial_over_a_chip_axis_crosses():
    source = _distributed(
        fm.SBP.broadcast(), fm.SBP.broadcast(), partial=fm.SBP.partial((0,)),
    )

    assert fm.boxing_crossing_chip_axes(source, TENSOR) == (0,)


def test_partial_retained_on_the_chip_axis_stays_chip_local():
    source = _distributed(
        fm.SBP.broadcast(), fm.SBP.broadcast(), partial=fm.SBP.partial((0, 1)),
    )
    target = _distributed(
        fm.SBP.broadcast(), fm.SBP.broadcast(), partial=fm.SBP.partial((0,)),
    )

    # Reducing away block axis 1 leaves the c-axis partial in place.
    assert fm.boxing_crossing_chip_axes(source, target) == ()


def test_exclusive_transition_over_a_chip_axis_crosses():
    source = _distributed(
        fm.SBP.broadcast(), fm.SBP.broadcast(), exclusive=fm.SBP.exclusive((0,)),
    )

    assert fm.boxing_crossing_chip_axes(source, TENSOR) == (0,)


def test_unit_chip_axis_never_crosses():
    placement = fm.Placement((1, 2, 4), "cyx", "cbb")
    source = fm.DistributedType(
        TENSOR,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 64)),
        placement,
    )

    assert fm.boxing_crossing_chip_axes(source, TENSOR) == ()


def test_block_cyclic_over_chip_and_block_axes_is_conservative():
    # Chip c owns two interleaved cyclic blocks per period, so the union over
    # its blocks is strided rather than one contiguous chip region.
    source = _distributed(fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0, 1), 8))
    target = _distributed(fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 16))

    assert fm.boxing_crossing_chip_axes(source, target) == (0,)


def test_block_cyclic_refinement_below_a_chip_split_stays_chip_local():
    source = _distributed(fm.SBP.broadcast(), fm.SBP.split(
        fm.SplitStage.contiguous((0,), 16), fm.SplitStage.block_cyclic((1,), 4),
    ))
    target = _distributed(fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 16))

    # The cyclic refinement stays inside chip c's own 16 contiguous elements.
    assert fm.boxing_crossing_chip_axes(source, target) == ()


def test_block_major_chip_minor_split_is_conservative():
    source = _distributed(fm.SBP.broadcast(), fm.SBP.split_contiguous((1, 0), 8))
    target = _distributed(fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 32))

    # Chip c's region is strided inside each block's shard, so the union over
    # blocks is not one contiguous chip region and containment is unproven.
    assert fm.boxing_crossing_chip_axes(source, target) == (0,)


def test_d_axis_split_crosses():
    placement = fm.Placement((2, 2, 4), "dyx", "dbb")
    source = fm.DistributedType(
        TENSOR,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 32)),
        placement,
    )

    assert fm.boxing_crossing_chip_axes(source, TENSOR) == (0,)
