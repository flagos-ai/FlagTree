# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from itertools import product

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.local_shard import local_shard_descriptor


@pytest.mark.parametrize("extent,block,expected", [(256, 1, 128), (257, 1, 128), (256, 2, 1), (512, 2, None), (7, 1, 128)])
def test_affine_stride_proves_every_active_owner_coordinate(extent, block, expected):
    placement = fm.Placement((8, 16), "yx", "bb")
    value = fm.DistributedType(
        fm.tensor_type("float32", (extent,)),
        (fm.SBP.split_block_cyclic((0, 1), block),), placement,
    )
    for owner in product(range(8), range(16)):
        axis = local_shard_descriptor(value, owner).axes[0]
        stride = axis.affine_stride
        if axis.active_extent.fixed_value > 1:
            assert stride == expected
        if stride is not None:
            origin = axis.map_local_to_global(0).fixed_value
            assert [axis.map_local_to_global(i).fixed_value for i in range(axis.active_extent.fixed_value)] == [
                origin + stride * i for i in range(axis.active_extent.fixed_value)
            ]


def test_affine_stride_composes_staged_cyclic_and_contiguous_splits():
    placement = fm.Placement((2, 4), "yx", "bb")
    policy = fm.SBPSplit((
        fm.SplitStage((0,), fm.BlockCyclicSplit(1)),
        fm.SplitStage((1,), fm.ContiguousSplit(4)),
    ))
    value = fm.DistributedType(fm.tensor_type("float32", (32,)), (policy,), placement)
    for owner in product(range(2), range(4)):
        axis = local_shard_descriptor(value, owner).axes[0]
        assert axis.affine_stride == 2
        assert [axis.map_local_to_global(i).fixed_value for i in range(4)] == [
            owner[0] + owner[1] * 8 + 2 * i for i in range(4)
        ]


@pytest.mark.parametrize("policy,stride", [(fm.SBP.broadcast(), 1), (fm.SBP.split_contiguous((0,)), 1), (fm.SBP.split_block_cyclic((0,), 1), 8)])
def test_linear_stage_stride_does_not_require_a_static_extent_or_owner(policy, stride):
    value = fm.DistributedType(fm.tensor_type("float32", (fm.DimVar("n", 1, 4096),)),
        (policy,), fm.Placement((8,), "x", "b"))
    axis = local_shard_descriptor(value, (fm.DimVar("owner", 0, 7),)).axes[0]
    assert axis.affine_stride == stride


def test_dynamic_multi_block_domain_does_not_get_a_false_affine_proof():
    value = fm.DistributedType(fm.tensor_type("float32", (fm.DimVar("n", 1, 4096),)),
        (fm.SBP.split_block_cyclic((0,), 2),), fm.Placement((8,), "x", "b"))
    assert local_shard_descriptor(value, (0,)).axes[0].affine_stride is None
