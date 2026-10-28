# Copyright 2026- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The k_major_n8_k16 K-contiguity probe only claims runs it can prove."""

import pytest

from triton.flagmega.codegen.triton.kernel_call_renderers import _k_major_n8_k16_k_contiguity

_TYPED = {"scalar_lane_shape": (8, 2, 8), "coordinate_space": "canonical_global"}
_SCALAR = {
    "logical_shape": (16, 1187, 8, 16),
    "scalar_storage_strides": (151936, 128, 16, 1),
    "coordinate_space": "canonical_global",
}


def _source(expression, capacity=256, hierarchy=(8, 16)):
    return {
        "local_capacity_shape": (capacity,),
        "scalar_lane_count": 1,
        "logical_coordinate_expressions": (expression,),
        "distributed_type": {"placement": {"hierarchy": hierarchy}},
    }


@pytest.mark.parametrize("weight,expression,capacity,block_k,expected", [
    # lm_head on the 8x16 mesh: K block-cyclic in 32-wide chunks over axis 0.
    (_TYPED, "shard_coord_0 * 32 + (local_coord_0 // 32) * 256 + local_coord_0 % 32", 256, 64, 16),
    (_SCALAR, "shard_coord_0 * 32 + (local_coord_0 // 32) * 256 + local_coord_0 % 32", 256, 64, 16),
    # Contiguous K split.
    (_TYPED, "shard_coord_0 * 256 + local_coord_0", 256, 64, 16),
    # 8-wide chunks break 16-element runs.
    (_TYPED, "shard_coord_0 * 8 + (local_coord_0 // 8) * 64 + local_coord_0 % 8", 256, 64, 1),
    # Contiguous but owner starts are not 16-aligned.
    (_TYPED, "shard_coord_0 * 24 + local_coord_0", 256, 64, 1),
    # Tiles that do not cover whole 16-element groups.
    (_TYPED, "shard_coord_0 * 256 + local_coord_0", 248, 64, 1),
    (_TYPED, "shard_coord_0 * 256 + local_coord_0", 256, 8, 1),
    # A non-unit payload stride is not contiguous storage.
    ({**_SCALAR, "scalar_storage_strides": (151936, 128, 32, 2)},
     "shard_coord_0 * 256 + local_coord_0", 256, 64, 1),
    # Unknown or non-arithmetic mappings never claim contiguity.
    (_TYPED, "some_helper(local_coord_0)", 256, 64, 1),
])
def test_k_contiguity_probe(weight, expression, capacity, block_k, expected):
    assert _k_major_n8_k16_k_contiguity(
        weight, _source(expression, capacity), local_k_capacity=capacity, block_k=block_k,
    ) == expected


def test_owner_local_storage_is_contiguous_by_construction():
    weight = {**_TYPED, "coordinate_space": "local"}
    assert _k_major_n8_k16_k_contiguity(
        weight, _source("unused"), local_k_capacity=256, block_k=64,
    ) == 16
