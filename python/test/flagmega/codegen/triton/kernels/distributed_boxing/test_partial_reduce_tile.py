# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The Boxing tile budget covers value lanes AND parallel partial owners."""

import pytest

from triton.flagmega.codegen.triton.kernel_call_renderers import prepare_kernel_calls
from .test_partial_reduce_local_abi import _abi, _raw


@pytest.mark.parametrize("values,lanes,budget,value_tile,owner_tile", [
    (1, 1, 1024, 1, 128), (3, 1, 1024, 4, 128),
    (17, 1, 1024, 32, 32), (65, 1, 1024, 128, 8),
    (1025, 1, 1024, 1024, 1), (3, 8, 1024, 32, 32),
    (1, 1, 16, 1, 16), (3, 8, 16, 16, 1),
])
def test_partial_reduction_shares_bounded_scalar_tile_between_two_axes(
    values, lanes, budget, value_tile, owner_tile,
):
    source = _abi(
        (values,), lane_count=lanes, storage_kind="compact_per_owner",
        coordinate_space="local", partial_axes=(0, 1), owner_stride=values * lanes,
    )
    raw = _raw((source,), (_abi((values,), lane_count=lanes),), ("gather_reduce_scatter",))
    raw["parameters"]["tile"] = budget
    leaf = prepare_kernel_calls((raw,), function_name="main")[0]["leaves"][0]
    assert leaf["tile"] == value_tile
    assert leaf["partial_owner_tile"] == owner_tile
    assert value_tile * owner_tile <= budget


def test_non_power_of_two_owner_group_has_masked_power_of_two_tile():
    source = _abi(
        (1,), storage_kind="compact_per_owner", coordinate_space="local",
        partial_axes=(0, 1), owner_stride=1,
    )
    source["distributed_type"]["placement"]["hierarchy"] = (3, 5)
    raw = _raw((source,), (_abi((1,)),), ("gather_reduce_scatter",))
    raw["parameters"]["tile"] = 1024
    leaf = prepare_kernel_calls((raw,), function_name="main")[0]["leaves"][0]
    assert leaf["partial_owner_count"] == 15
    assert leaf["partial_owner_tile"] == 16
