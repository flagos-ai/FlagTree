# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm


def _fixed_shape(value):
    return tuple(dimension.fixed_value for dimension in fm.local_shape(value))


def test_contiguous_split_composes_across_two_mesh_axes():
    placement = fm.Placement((2, 4), "yx", "bb")
    value = fm.DistributedType(
        fm.tensor_type("bfloat16", (64, 32)),
        (fm.SBP.split_contiguous((0, 1)), fm.SBP.broadcast()),
        placement,
    )

    assert _fixed_shape(value) == (8, 32)
    assert fm.placement_owner_count(value) == 8
    assert fm.is_fully_sharded_across_placement(value)


def test_sequential_contiguous_and_block_cyclic_stages_reserve_largest_component():
    placement = fm.Placement((2, 4), "yx", "bb")
    value = fm.DistributedType(
        fm.tensor_type("bfloat16", (1000, 32)),
        (
            fm.SBP.split(
                fm.SplitStage.contiguous((0,)),
                fm.SplitStage.block_cyclic((1,), 64),
            ),
            fm.SBP.broadcast(),
        ),
        placement,
    )

    # 1000 -> ceil(1000/2)=500 -> ceil(ceil(500/64)/4)*64=128.
    assert _fixed_shape(value) == (128, 32)


def test_partial_replica_keeps_dense_component_shape_but_requires_owner_storage():
    placement = fm.Placement((2, 4), "yx", "bb")
    tensor = fm.tensor_type("float32", (17, 9))
    value = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
        partial=fm.SBP.partial((0, 1)),
    )

    assert _fixed_shape(value) == (17, 9)
    assert not fm.is_fully_sharded_across_placement(value)


def test_contiguous_split_uses_explicit_nncase_local_capacity():
    value = fm.DistributedType(
        fm.tensor_type("bfloat16", (12,)),
        (fm.SBP.split_contiguous((0,), granularity=5),),
        fm.Placement((3,), "x", "b"),
    )

    assert _fixed_shape(value) == (5,)
