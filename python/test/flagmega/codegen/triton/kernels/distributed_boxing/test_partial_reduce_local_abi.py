# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.codegen.triton.kernel_call_renderers import (
    prepare_kernel_calls,
)
from triton.flagmega.errors import CodegenError


def _abi(
    shape,
    *,
    local_shape=None,
    storage_kind="canonical_global",
    coordinate_space="canonical_global",
    coordinates=None,
    axis_policies=None,
    partial_axes=None,
    owner_stride=0,
    lane_count=1,
):
    local_shape = tuple(shape if local_shape is None else local_shape)
    scalar_strides = []
    stride = lane_count
    for extent in reversed(local_shape):
        scalar_strides.append(stride)
        stride *= extent
    return {
        "storage": "input",
        "storage_kind": storage_kind,
        "pool_byte_offset": 0,
        "scalar_dtype": "bfloat16",
        "scalar_itemsize": 2,
        "logical_shape": tuple(shape),
        "local_capacity_shape": local_shape,
        "active_shape_expressions": tuple(str(value) for value in local_shape),
        "logical_coordinate_expressions": tuple(
            coordinates
            or (f"local_coord_{axis}" for axis in range(len(local_shape)))
        ),
        "scalar_storage_strides": tuple(reversed(scalar_strides)),
        "scalar_lane_shape": () if lane_count == 1 else (lane_count,),
        "scalar_lane_count": lane_count,
        "component_stride_scalar_elements": owner_stride,
        "coordinate_space": coordinate_space,
        "distributed_type": {
            "placement": {"hierarchy": (8, 16)},
            "axis_policies": tuple(
                axis_policies
                or ({"kind": "broadcast"} for _ in shape)
            ),
            "partial": (
                None
                if partial_axes is None
                else {"axes": tuple(partial_axes), "reduce_op": "sum"}
            ),
        },
    }


def _binding(name, abi):
    return {
        "formal": name,
        "actual": name,
        "runtime_argument": name,
        "runtime_value_kind": "pointer",
        "abi": abi,
    }


def _raw(sources, results, transitions):
    return {
        "call": "boxing",
        "family": "distributed_boxing",
        "variant": "gather_reduce_scatter",
        "execution_kind": "collective",
        "parameters": {
            "tile": 16,
            "leaf_transitions": tuple(transitions),
        },
        "semantic_attrs": {},
        "inputs": ({
            "formal": "value",
            "buffers": tuple(
                _binding(f"source_{index}", abi)
                for index, abi in enumerate(sources)
            ),
        },),
        "outputs": tuple(
            {
                "formal": "result" if len(results) == 1 else f"result_{index}",
                "buffers": (_binding(f"result_{index}", abi),),
            }
            for index, abi in enumerate(results)
        ),
        "workspaces": (),
    }


def _prepare(sources, results, transitions):
    return prepare_kernel_calls(
        (_raw(sources, results, transitions),), function_name="main"
    )[0]


def test_all_axis_partial_reduces_compact_owner_components_once():
    source = _abi(
        (1, 8),
        storage_kind="compact_per_owner",
        coordinate_space="local",
        partial_axes=(0, 1),
        owner_stride=8,
    )
    result = _abi((1, 8))

    leaf = _prepare(
        (source,), (result,), ("gather_reduce_scatter",)
    )["leaves"][0]

    assert leaf["reduction"] is True
    assert leaf["capacity"] == 8
    assert leaf["partial_owner_count"] == 128
    assert leaf["partial_owner"] == "(boxing_partial_member)"
    assert leaf["owner_stride"] == 8
    assert leaf["writer_active"] == (
        "(shard_y == 0) & (shard_x == 0)"
    )


def test_hybrid_split_and_partial_reduce_preserves_local_logical_shard():
    source = _abi(
        (1, 2048),
        local_shape=(1, 256),
        storage_kind="compact_per_owner",
        coordinate_space="local",
        coordinates=("local_coord_0", "local_coord_1 + shard_coord_0 * 256"),
        axis_policies=(
            {"kind": "broadcast"},
            {
                "kind": "split",
                "stages": ({"hierarchy_axes": (0,)},),
            },
        ),
        partial_axes=(1,),
        owner_stride=256,
    )
    result = _abi((1, 2048))

    leaf = _prepare(
        (source,), (result,), ("gather_reduce_scatter",)
    )["leaves"][0]

    assert leaf["capacity"] == 256
    assert leaf["partial_owner_count"] == 16
    assert leaf["partial_owner"] == (
        "(shard_y * 16 + (boxing_partial_member))"
    )
    assert leaf["writer_active"] == "(shard_x == 0)"
    assert "shard_y" in leaf["result_offset"]


def test_partial_y_reduction_keeps_every_distinct_x_output_writer():
    source = _abi(
        (1, 256),
        local_shape=(1, 16),
        storage_kind="compact_per_owner",
        coordinate_space="local",
        coordinates=("local_coord_0", "local_coord_1 + shard_coord_1 * 16"),
        axis_policies=(
            {"kind": "broadcast"},
            {
                "kind": "split",
                "stages": ({"hierarchy_axes": (1,)},),
            },
        ),
        partial_axes=(0,),
        owner_stride=128,
        lane_count=8,
    )
    result = _abi(
        (1, 256),
        local_shape=(1, 16),
        coordinates=("local_coord_0", "local_coord_1 + shard_coord_1 * 16"),
        axis_policies=(
            {"kind": "broadcast"},
            {
                "kind": "split",
                "stages": ({"hierarchy_axes": (1,)},),
            },
        ),
        lane_count=8,
    )

    leaf = _prepare(
        (source,), (result,), ("gather_reduce_scatter",)
    )["leaves"][0]

    assert leaf["partial_owner_count"] == 8
    assert leaf["partial_owner"] == (
        "((boxing_partial_member) * 16 + shard_x)"
    )
    assert leaf["writer_active"] == "(shard_y == 0)"
    assert "shard_x" in leaf["result_offset"]


def test_tuple_boxing_retains_identity_and_partial_leaves():
    local = _abi((1, 4), storage_kind="compact_local", coordinate_space="local")
    partial = _abi(
        (1, 8),
        storage_kind="compact_per_owner",
        coordinate_space="local",
        partial_axes=(0, 1),
        owner_stride=8,
    )
    result = _abi((1, 8))

    call = _prepare(
        (local, partial),
        (local, result),
        ("identity", "gather_reduce_scatter"),
    )

    assert len(call["leaves"]) == 2
    assert call["leaves"][0]["reduction"] is False
    assert call["leaves"][0]["mode"] == "local_copy"
    assert call["leaves"][1]["reduction"] is True
    assert "capacity" not in call


def test_partial_vector_leaf_expands_physical_elements_into_scalar_lanes():
    source = _abi(
        (1, 2),
        storage_kind="compact_per_owner",
        coordinate_space="local",
        partial_axes=(0, 1),
        owner_stride=8,
        lane_count=4,
    )
    result = _abi((1, 2), lane_count=4)

    leaf = _prepare(
        (source,), (result,), ("gather_reduce_scatter",)
    )["leaves"][0]

    assert leaf["capacity"] == 8
    assert "// 4" in leaf["source_offset"]
    assert "% 4" in leaf["source_offset"]
    assert "% 4" in leaf["result_offset"]


def test_partial_canonical_storage_is_rejected_instead_of_double_counted():
    source = _abi((1, 8), partial_axes=(0, 1))
    result = _abi((1, 8))

    with pytest.raises(CodegenError, match="compact source component"):
        _prepare((source,), (result,), ("gather_reduce_scatter",))


def test_leaf_transition_count_is_part_of_the_executable_contract():
    value = _abi((1, 8))

    with pytest.raises(CodegenError, match="leaf transitions must match"):
        _prepare((value,), (value,), ())
