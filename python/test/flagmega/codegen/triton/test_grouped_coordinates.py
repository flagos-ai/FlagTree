# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Group proofs include all owners and preserve scalar-coordinate units."""

import pytest
from itertools import product

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.grouped_coordinates import tiles_stay_within_groups


def distributed(size, policy, mesh=(2, 4)):
    return fm.DistributedType(fm.tensor_type("bfloat16", (1, size)),
                              (fm.SBP.broadcast(), policy), fm.Placement(mesh, "yx", "bb"))


@pytest.mark.parametrize("tile", [1, 2, 4, 8, 16, 32])
def test_contiguous_aligned_shards_have_uniform_groups(tile):
    value = distributed(4096, fm.SBP.split_contiguous((0, 1)))
    assert tiles_stay_within_groups(value, -1, tile, 128)


@pytest.mark.parametrize("size,tile,group", [(24, 4, 6), (48, 4, 12), (136, 16, 128)])
def test_crossing_groups_or_unaligned_owner_origins_are_not_uniform(size, tile, group):
    value = distributed(size, fm.SBP.split_contiguous((0, 1)))
    assert not tiles_stay_within_groups(value, -1, tile, group)


@pytest.mark.parametrize("policy", [fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0, 1), 8)])
def test_aligned_short_final_tile_is_allowed(policy):
    value = distributed(61, policy)
    assert tiles_stay_within_groups(value, -1, 4, 16)


def test_cyclic_non_unit_step_is_not_a_uniform_group_proof():
    value = distributed(4096, fm.SBP.split_block_cyclic((0, 1), 4))
    assert not tiles_stay_within_groups(value, -1, 8, 128)


def test_single_cyclic_block_per_owner_can_be_proved():
    value = distributed(32, fm.SBP.split_block_cyclic((0, 1), 4))
    assert tiles_stay_within_groups(value, -1, 4, 16)


def test_local_tensor_is_aligned_but_dynamic_extent_is_not_assumed():
    assert tiles_stay_within_groups(fm.tensor_type("bfloat16", (1, 256)), -1, 16, 128)
    assert not tiles_stay_within_groups(fm.tensor_type("bfloat16", (1, "values")), -1, 16, 128)


def test_vector_element_indices_are_not_scalar_head_indices():
    value = fm.tensor_type(fm.VectorType(fm.DType.BFLOAT16, (4,)), (1, 256))
    assert not tiles_stay_within_groups(value, -1, 16, 128)


@pytest.mark.parametrize("policy", [fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1)),
                                   fm.SBP.split_block_cyclic((0, 1), 1),
                                   fm.SBP.split_block_cyclic((0, 1), 4)])
def test_every_accepted_proof_agrees_with_complete_small_coordinate_domains(policy):
    for size, tile, group in product((8, 24, 32, 48, 64), (1, 2, 4, 8), (6, 12, 16, 32)):
        value = distributed(size, policy)
        if not tiles_stay_within_groups(value, -1, tile, group):
            continue
        for owner in product(range(2), range(4)):
            axis = fm.local_shard_descriptor(value, owner).axes[-1]
            for start in range(0, axis.active_extent.fixed_value, tile):
                groups = {axis.map_local_to_global(index).fixed_value // group
                          for index in range(start, min(start + tile, axis.active_extent.fixed_value))}
                assert len(groups) == 1
