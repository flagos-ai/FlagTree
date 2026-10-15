# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A source owner selected per output must stay on the value axis of a load.

This is the routed-expert down-projection Boxing in the full GLM TP4 graph:
chip partials, expert owners on y, and feature owners on x. The destination
splits only features over y/x. Treating the per-value owner as the reduction
axis turns one partial into 256 partials and creates a 256x256 register tile.
"""

import numpy as np
import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.kernel_call_renderers import prepare_kernel_calls

from .test_partial_reduce_local_abi import _abi, _raw


def routed_expert_partial_leaf():
    placement = fm.Placement((4, 8, 16), "cyx", "cbb")
    tensor = fm.tensor_type("float32", (1, 8, 4096))
    source_type = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 1), fm.SBP.split_contiguous((2,), 256)),
        placement, partial=fm.SBP.partial((0,)),
    )
    result_type = fm.DistributedType(
        tensor, (fm.SBP.broadcast(), fm.SBP.broadcast(), fm.SBP.split_contiguous((1, 2), 32)), placement,
    )
    source = _abi(
        (1, 8, 4096), local_shape=(1, 1, 256), storage_kind="compact_per_owner", coordinate_space="local",
        owner_stride=256, partial_axes=(0,), hierarchy=(4, 8, 16), hierarchy_levels="cbb",
        coordinates=("local_coord_0", "local_coord_1 + shard_coord_1", "local_coord_2 + shard_coord_2 * 256"),
    )
    result = _abi(
        (1, 8, 4096), local_shape=(1, 8, 32), hierarchy=(4, 8, 16), hierarchy_levels="cbb",
        coordinates=("local_coord_0", "local_coord_1", "local_coord_2 + shard_coord_2 * 32 + shard_coord_1 * 512"),
    )
    for abi, value_type in ((source, source_type), (result, result_type)):
        abi.update(scalar_dtype="float32", scalar_itemsize=4, memory_space="workspace", storage="workspace",
                   distributed_type=value_type.to_data())
    result["scalar_storage_strides"] = (32768, 4096, 1)
    raw = _raw((source,), (result,), ("gather_reduce_scatter",))
    raw["parameters"]["tile"] = 256
    return prepare_kernel_calls((raw,), function_name="main")[0]["leaves"][0]


@pytest.mark.parametrize("y,x", [(0, 0), (0, 15), (7, 15), (3, 5)])
def test_partial_source_addresses_preserve_value_and_reduction_axes(y, x):
    leaf = routed_expert_partial_leaf()
    offsets = np.arange(leaf["tile"])
    assert leaf["partial_owner_count"] == leaf["partial_owner_tile"] == 1
    for chip in range(4):
        namespace = {"boxing_offsets": offsets, "boxing_partial_member": np.arange(leaf["partial_owner_tile"]),
                     "boxing_peer": chip, "shard_coord0": chip, "shard_coord1": y, "shard_coord2": x}
        owners = np.asarray(eval(leaf["partial_owner"], {"__builtins__": {}}, namespace))
        local = np.asarray(eval(leaf["source_offset"], {"__builtins__": {}}, namespace))
        addresses = owners[:, None] * leaf["owner_stride"] + local[None, :]
        assert addresses.shape == (1, 256), "per-value owners became a second value axis"
        feature = offsets % 32 + (y * 16 + x) * 32
        expected_owner = (chip * 8 + offsets // 32) * 16 + feature // 256
        np.testing.assert_array_equal(addresses[0], expected_owner * 256 + feature % 256)
