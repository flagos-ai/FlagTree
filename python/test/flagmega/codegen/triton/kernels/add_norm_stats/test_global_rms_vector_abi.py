# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from itertools import product
from math import prod

import pytest

from triton.flagmega.codegen.triton.kernel_call_renderers import _add_norm_stats_call
from triton.flagmega.errors import CodegenError


def _abi(shape, *, dtype="bfloat16", lanes=()):
    lane_count = 1
    for lane in lanes:
        lane_count *= lane
    return {
        "storage": "workspace",
        "storage_kind": "replicated_global",
        "pool_byte_offset": 0,
        "scalar_dtype": dtype,
        "scalar_itemsize": 4 if dtype == "float32" else 2,
        "logical_shape": tuple(shape),
        "local_capacity_shape": tuple(shape),
        "active_shape_expressions": tuple(str(value) for value in shape),
        "logical_coordinate_expressions": tuple(
            f"coord_{axis}" for axis in range(len(shape))
        ),
        "scalar_storage_strides": (),
        "scalar_lane_shape": tuple(lanes),
        "scalar_lane_count": lane_count,
        "component_stride_scalar_elements": 1,
        "coordinate_space": "global",
        "distributed_type": {
            "placement": {"hierarchy": (8, 16)},
            "axis_policies": tuple({"kind": "broadcast"} for _ in shape),
            "partial": None,
        },
    }


def _parameter(name, abi):
    return {
        "formal": name,
        "buffers": ({
            "formal": name,
            "actual": name,
            "runtime_argument": name,
            "abi": abi,
        },),
    }


def _raw(*, residual_lanes=(8,)):
    value = _abi((1, 2), lanes=(8,))
    stats = _abi((1, 1, 1), dtype="float32")
    partials = _abi((128,), dtype="float32")
    return {
        "semantic_attrs": {"axis": 1, "use_mean": False},
        "parameters": {
            "family": "add_norm_stats",
            "variant": "rms",
            "tile": 16,
            "owner_count": 128,
        },
        "inputs": (
            _parameter("input", value),
            _parameter("addend", _abi((1, 2), lanes=residual_lanes)),
        ),
        "outputs": (
            _parameter("result_0", value),
            _parameter("result_1", stats),
        ),
        "workspaces": (_parameter("norm_stats_partials", partials),),
    }


def test_global_rms_counts_vector_lanes_in_the_scalar_iteration_domain():
    call = _add_norm_stats_call(_raw())

    assert call["mode"] == "global_rms"
    assert call["elements"] == 16


@pytest.mark.parametrize("hierarchy,levels", [
    ((4, 2, 8), "cbb"),
    ((2, 4, 8), "bcb"),
    ((2, 3, 4, 2), "cdbb"),
    ((2, 4, 8), "bbb"),
])
def test_global_rms_partitions_stats_within_each_workspace_replica(hierarchy, levels):
    raw = _raw()
    raw["parameters"]["owner_count"] = prod(hierarchy)
    for parameter in (*raw["inputs"], *raw["outputs"]):
        abi = parameter["buffers"][0]["abi"]
        abi["storage_kind"] = "canonical_global"
        abi["distributed_type"]["placement"] = {
            "hierarchy": hierarchy, "hierarchy_levels": levels,
        }
    call = _add_norm_stats_call(raw)
    block_axes = [axis for axis, level in enumerate(levels) if level == "b"]
    block_owner_count = prod(hierarchy[axis] for axis in block_axes)
    assert call["owner_count"] == block_owner_count
    for coordinates in product(*(range(size) for size in hierarchy)):
        expected = global_index = 0
        for axis, coordinate in enumerate(coordinates):
            global_index = global_index * hierarchy[axis] + coordinate
            if axis in block_axes:
                expected = expected * hierarchy[axis] + coordinate
        actual = eval(call["owner_index"], {"__builtins__": {}}, {
            "shard_index": global_index,
            **{f"shard_coord{axis}": coord for axis, coord in enumerate(coordinates)},
        })
        assert actual == expected


def test_global_rms_rejects_mismatched_vector_lane_abis():
    with pytest.raises(CodegenError, match="vector lanes must match"):
        _add_norm_stats_call(_raw(residual_lanes=(4,)))
