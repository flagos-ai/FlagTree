# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.codegen.triton.kernel_call_renderers import (
    _add_norm_stats_call,
)
from triton.flagmega.errors import CodegenError


def _abi(
    shape,
    *,
    scalar_dtype="bfloat16",
    storage_kind="compact_per_owner",
    coordinate_space="local",
    coordinates=None,
    partial_axes=None,
):
    strides = []
    stride = 1
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return {
        "storage": "workspace",
        "storage_kind": storage_kind,
        "pool_byte_offset": 0,
        "scalar_dtype": scalar_dtype,
        "scalar_itemsize": 4 if scalar_dtype == "float32" else 2,
        "logical_shape": tuple(shape),
        "local_capacity_shape": tuple(shape),
        "active_shape_expressions": tuple(str(value) for value in shape),
        "logical_coordinate_expressions": tuple(
            coordinates
            or (f"local_coord_{axis}" for axis in range(len(shape)))
        ),
        "scalar_storage_strides": tuple(reversed(strides)),
        "scalar_lane_shape": (),
        "scalar_lane_count": 1,
        "component_stride_scalar_elements": stride,
        "coordinate_space": coordinate_space,
        "distributed_type": {
            "placement": {"hierarchy": (8, 16)},
            "axis_policies": tuple(
                {"kind": "broadcast"} for _ in shape
            ),
            "partial": (
                None
                if partial_axes is None
                else {"axes": tuple(partial_axes), "reduce_op": "sum"}
            ),
        },
    }


def _parameter(formal, abi):
    return {
        "formal": formal,
        "buffers": ({
            "formal": formal,
            "actual": formal,
            "runtime_argument": formal,
            "abi": abi,
        },),
    }


def _raw(*, stats_partial=(0, 1), residual_coordinates=None):
    value_coordinates = (
        "local_coord_0",
        "local_coord_1 + (shard_y * 16 + shard_x) * 16",
    )
    source = _abi((1, 16), coordinates=value_coordinates)
    residual = _abi(
        (1, 16),
        coordinates=residual_coordinates or value_coordinates,
    )
    result = _abi((1, 16), coordinates=value_coordinates)
    stats = _abi(
        (1, 1, 1), scalar_dtype="float32", partial_axes=stats_partial
    )
    return {
        "semantic_attrs": {"axis": -1, "use_mean": False},
        "parameters": {
            "family": "add_norm_stats",
            "variant": "local_partial_rms",
            "tile": 128,
            "owner_count": 128,
        },
        "inputs": (
            _parameter("input", source),
            _parameter("addend", residual),
        ),
        "outputs": (
            _parameter("result_0", result),
            _parameter("result_1", stats),
        ),
        "workspaces": (),
    }


def test_local_partial_rms_uses_only_owner_local_offsets_and_one_partial_stat():
    call = _add_norm_stats_call(_raw())

    assert call["mode"] == "local_partial_rms"
    assert call["local_capacity"] == 16
    assert call["tile"] == 16
    assert call["stats_writer_active"] == "True"
    assert "shard_" not in call["source_offset"]
    assert "shard_" not in call["residual_offset"]
    assert "shard_" not in call["result_offset"]
    assert "shard_" not in call["stats_offset"]
    assert set(call["stats_offset"]) <= set("01() *+")
    assert call["output_type"] == "tl.bfloat16"


def test_local_partial_rms_rejects_a_materialized_statistics_result():
    with pytest.raises(CodegenError, match="explicit Sum-partial"):
        _add_norm_stats_call(_raw(stats_partial=None))


def test_local_partial_rms_rejects_mismatched_local_value_mappings():
    with pytest.raises(CodegenError, match="explicit Boxing"):
        _add_norm_stats_call(_raw(residual_coordinates=(
            "local_coord_0",
            "local_coord_1 + shard_x * 16",
        )))
