# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A partial Boxing publishes its result in every chip's own destination pool.

Chip and node axes do not share one allocation: every chip owns a private
instance of the destination pool at the same offset.  Electing one writer
across a chip axis therefore leaves the other chips' instances untouched, so a
reduction whose destination is replicated over chips must keep one writer per
chip rather than one writer per logical value.
"""

from .test_partial_reduce_local_abi import _abi, _prepare


def test_chip_replicated_destination_keeps_one_writer_per_chip():
    source = _abi(
        (1, 8),
        storage_kind="compact_per_owner",
        coordinate_space="local",
        partial_axes=(1,),
        owner_stride=8,
        hierarchy=(4, 8),
        hierarchy_levels="cb",
    )
    result = _abi((1, 8), hierarchy=(4, 8), hierarchy_levels="cb")

    leaf = _prepare((source,), (result,), ("gather_reduce_scatter",))["leaves"][0]

    # The chip axis is a redundant writer axis for one shared allocation, but
    # each chip holds its own destination instance, so it must stay a writer.
    assert leaf["writer_active"] == "(shard_x == 0)"


def test_chip_partial_axis_still_elects_one_writer_per_chip():
    """The reduction consumes the chip axis; the destination still needs it."""

    source = _abi(
        (1, 4),
        storage_kind="compact_per_owner",
        coordinate_space="local",
        partial_axes=(0,),
        owner_stride=4,
        hierarchy=(4, 8),
        hierarchy_levels="cb",
    )
    result = _abi((1, 4), hierarchy=(4, 8), hierarchy_levels="cb")

    leaf = _prepare((source,), (result,), ("gather_reduce_scatter",))["leaves"][0]

    assert leaf["writer_active"] == "(shard_x == 0)"


def test_block_only_placement_is_unchanged():
    source = _abi(
        (1, 8),
        storage_kind="compact_per_owner",
        coordinate_space="local",
        partial_axes=(0, 1),
        owner_stride=8,
        hierarchy=(8, 16),
        hierarchy_levels="bb",
    )
    result = _abi((1, 8), hierarchy=(8, 16), hierarchy_levels="bb")

    leaf = _prepare((source,), (result,), ("gather_reduce_scatter",))["leaves"][0]

    assert leaf["writer_active"] == "(shard_y == 0) & (shard_x == 0)"
