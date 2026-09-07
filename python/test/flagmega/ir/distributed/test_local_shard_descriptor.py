# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm


def _fixed(values):
    return tuple(value.fixed_value for value in values)


def test_contiguous_shard_exposes_capacity_active_region_and_global_offset():
    distributed = fm.DistributedType(
        fm.tensor_type("bfloat16", (12, 3)),
        (fm.SBP.split_contiguous((0,), granularity=5), fm.SBP.broadcast()),
        fm.Placement((3,), "x", "b"),
    )

    first = fm.local_shard_descriptor(distributed, (0,))
    tail = fm.local_shard_descriptor(distributed, (2,))

    assert _fixed(first.local_capacity_shape) == (5, 3)
    assert _fixed(first.active_shape) == (5, 3)
    assert first.contiguous_region is not None
    assert _fixed(first.contiguous_region.offset) == (0, 0)
    assert _fixed(tail.local_capacity_shape) == (5, 3)
    assert _fixed(tail.active_shape) == (2, 3)
    assert tail.contiguous_region is not None
    assert _fixed(tail.contiguous_region.offset) == (10, 0)


def test_block_cyclic_shard_maps_dense_local_coordinates_to_global_coordinates():
    distributed = fm.DistributedType(
        fm.tensor_type("bfloat16", (22,)),
        (fm.SBP.split_block_cyclic((0,), block_size=4),),
        fm.Placement((3,), "x", "b"),
    )

    middle = fm.local_shard_descriptor(distributed, (1,))
    tail = fm.local_shard_descriptor(distributed, (2,))

    assert _fixed(middle.local_capacity_shape) == (8,)
    assert _fixed(middle.active_shape) == (8,)
    assert tuple(middle.axes[0].map_local_to_global(index).fixed_value for index in range(8)) == (
        4, 5, 6, 7, 16, 17, 18, 19,
    )
    assert _fixed(tail.active_shape) == (6,)
    assert tuple(tail.axes[0].map_local_to_global(index).fixed_value for index in range(6)) == (
        8, 9, 10, 11, 20, 21,
    )
    assert middle.contiguous_region is None


def test_staged_split_uses_each_owner_active_extent_but_reserves_max_capacity():
    distributed = fm.DistributedType(
        fm.tensor_type("bfloat16", (20,)),
        (
            fm.SBP.split(
                fm.SplitStage.contiguous((0,), granularity=10),
                fm.SplitStage.block_cyclic((1,), block_size=4),
            ),
        ),
        fm.Placement((2, 2), "yx", "bb"),
    )

    owner_00 = fm.local_shard_descriptor(distributed, (0, 0))
    owner_11 = fm.local_shard_descriptor(distributed, (1, 1))

    assert _fixed(owner_00.local_capacity_shape) == (8,)
    assert _fixed(owner_00.active_shape) == (6,)
    assert tuple(owner_00.axes[0].map_local_to_global(index).fixed_value for index in range(6)) == (
        0, 1, 2, 3, 8, 9,
    )
    assert _fixed(owner_11.local_capacity_shape) == (8,)
    assert _fixed(owner_11.active_shape) == (4,)
    assert tuple(owner_11.axes[0].map_local_to_global(index).fixed_value for index in range(4)) == (
        14, 15, 16, 17,
    )


def test_partial_axes_do_not_change_local_tensor_shape():
    distributed = fm.DistributedType(
        fm.tensor_type("float32", (7, 5)),
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        fm.Placement((2, 4), "yx", "bb"),
        partial=fm.SBP.partial((0,)),
    )

    descriptor = fm.local_shard_descriptor(distributed, (1, 3))

    assert _fixed(descriptor.local_capacity_shape) == (7, 5)
    assert _fixed(descriptor.active_shape) == (7, 5)
    assert descriptor.partial_axes == (0,)
    assert descriptor.partial_group_coordinates == ((0, 3), (1, 3))
