# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Transfer tile width is a target bound, not a scalar/vector packing atom."""

import pytest

from triton.flagmega.codegen.triton.kernel_call_renderers import prepare_kernel_calls
from triton.flagmega.errors import CodegenError
from .test_partial_reduce_local_abi import _abi, _raw


@pytest.mark.parametrize("capacity,lanes,tile", [(1, 1, 1), (3, 1, 4), (31, 1, 32), (3, 8, 32), (257, 8, 1024), (6144, 1, 1024)])
@pytest.mark.parametrize("partial", [False, True])
def test_transfer_tiles_are_bounded_in_scalar_elements(capacity, lanes, tile, partial):
    source = _abi((1, capacity), lane_count=lanes,
        storage_kind="compact_per_owner" if partial else "canonical_global",
        coordinate_space="local" if partial else "canonical_global",
        partial_axes=(0, 1) if partial else None, owner_stride=capacity * lanes if partial else 0)
    raw = _raw((source,), (_abi((1, capacity), lane_count=lanes),),
        ("gather_reduce_scatter" if partial else "tensor_load",))
    raw["parameters"]["tile"] = 1024
    leaf = prepare_kernel_calls((raw,), function_name="main")[0]["leaves"][0]
    assert leaf["tile"] == tile
    assert leaf["capacity"] == capacity * lanes


@pytest.mark.parametrize("tile", [0, -1, 24, True])
def test_transfer_tile_edits_must_be_positive_powers_of_two(tile):
    raw = _raw((_abi((1, 6144)),), (_abi((1, 6144)),), ("tensor_load",))
    raw["parameters"]["tile"] = tile
    with pytest.raises(CodegenError, match="power of two"):
        prepare_kernel_calls((raw,), function_name="main")
