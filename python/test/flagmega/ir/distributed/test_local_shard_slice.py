# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""``LocalShardAxisDescriptor.local_slice`` must agree with the element map.

Offline materializers (rdata packing, ``MaterializeLocalShards``) select an
owner's elements once per owner per axis.  Proving the selection is a strided
slice removes a per-element symbolic ``map_local_to_global`` walk, so the proof
has to be exactly as strong as the element map it replaces.
"""

from itertools import product

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.local_shard import local_shard_descriptor


def _elements(axis):
    return [
        axis.map_local_to_global(index).fixed_value
        for index in range(axis.active_extent.fixed_value)
    ]


@pytest.mark.parametrize("policy,extent", [
    (fm.SBP.broadcast(), extent) for extent in (256, 257, 7)
] + [
    # A contiguous split needs a divisible extent; block-cyclic covers tails.
    (fm.SBP.split_contiguous((0, )), 256),
    (fm.SBP.split_contiguous((0, 1)), 256),
] + [
    (policy, extent)
    for policy in (fm.SBP.split_block_cyclic((0, ), 1),
                   fm.SBP.split_block_cyclic((0, 1), 1),
                   fm.SBP.split_block_cyclic((0, ), 2),
                   fm.SBP.split_block_cyclic((1, ), 4))
    for extent in (256, 257, 7)
])
def test_local_slice_selects_exactly_the_mapped_elements(policy, extent):
    placement = fm.Placement((8, 16), "yx", "bb")
    value = fm.DistributedType(
        fm.tensor_type("float32", (extent, )), (policy, ), placement)
    for owner in product(range(8), range(16)):
        axis = local_shard_descriptor(value, owner).axes[0]
        window = axis.local_slice
        if window is None:
            continue
        assert list(range(extent))[window] == _elements(axis)


def test_an_unsplit_axis_is_a_whole_axis_slice():
    """The common case: a broadcast axis must not cost an index list."""

    value = fm.DistributedType(
        fm.tensor_type("float32", (288, )), (fm.SBP.broadcast(), ),
        fm.Placement((4, ), "c", "c"))
    axis = local_shard_descriptor(value, (0, )).axes[0]
    assert axis.local_slice == slice(0, 288, 1)


def test_a_non_affine_axis_has_no_slice_proof():
    value = fm.DistributedType(
        fm.tensor_type("float32", (fm.DimVar("n", 1, 4096), )),
        (fm.SBP.split_block_cyclic((0, ), 2), ), fm.Placement((8, ), "x", "b"))
    assert local_shard_descriptor(value, (0, )).axes[0].local_slice is None


def test_a_dynamic_extent_has_no_slice_proof():
    """``affine_stride`` tolerates a dynamic extent; a slice cannot."""

    value = fm.DistributedType(
        fm.tensor_type("float32", (fm.DimVar("n", 1, 4096), )),
        (fm.SBP.split_contiguous((0, )), ), fm.Placement((8, ), "x", "b"))
    axis = local_shard_descriptor(value, (0, )).axes[0]
    assert axis.affine_stride == 1
    assert axis.local_slice is None
