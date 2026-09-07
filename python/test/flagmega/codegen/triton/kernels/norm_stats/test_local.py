# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.kernel_call_renderers import _norm_stats_call


def _abi(
    *,
    shape,
    strides,
    scalar_lane_count=1,
    scalar_lane_shape=(),
    scalar_dtype="float32",
):
    return {
        "storage": "input",
        "storage_kind": "compact_local",
        "pool_byte_offset": 0,
        "scalar_dtype": scalar_dtype,
        "scalar_itemsize": 4,
        "logical_shape": tuple(shape),
        "local_capacity_shape": tuple(shape),
        "active_shape_expressions": tuple(str(value) for value in shape),
        "logical_coordinate_expressions": tuple(
            f"local_coord_{axis}" for axis in range(len(shape))
        ),
        "scalar_storage_strides": tuple(strides),
        "scalar_lane_shape": tuple(scalar_lane_shape),
        "scalar_lane_count": scalar_lane_count,
        "component_stride_scalar_elements": 0,
        "coordinate_space": "local",
        "distributed_type": None,
    }


def _binding(formal, argument, abi):
    return {
        "formal": formal,
        "buffers": ({
            "formal": formal,
            "actual": formal,
            "runtime_argument": argument,
            "abi": abi,
        },),
    }


def test_non_last_axis_mean_stats_preserve_every_outer_coordinate():
    source = _abi(shape=(2, 3, 5), strides=(15, 5, 1))
    result = _abi(shape=(2, 2, 1, 1), strides=(2, 1, 1, 1))
    raw = {
        "inputs": (_binding("input", "source", source),),
        "outputs": (_binding("result", "result", result),),
        "workspaces": (),
        "semantic_attrs": {"axis": 1, "use_mean": True},
        "parameters": {"block_size": 128},
    }

    call = _norm_stats_call(raw)

    assert call["axis"] == 1
    assert call["use_mean"] is True
    assert call["outer_capacity"] == 2
    assert call["reduction_capacity"] == 15
    assert "norm_outer_index" in call["source_offset"]
    assert "norm_reduction_offsets" in call["source_offset"]
    assert len(call["result_offsets"]) == 2
    assert call["partial_stats"] is None


def test_vector_lanes_are_part_of_the_reduction_address_not_dropped():
    source = _abi(
        shape=(2, 3, 5),
        strides=(60, 20, 4),
        scalar_lane_count=4,
        scalar_lane_shape=(4,),
        scalar_dtype="bfloat16",
    )
    result = _abi(shape=(1, 2, 1, 1), strides=(2, 1, 1, 1))
    raw = {
        "inputs": (_binding("input", "source", source),),
        "outputs": (_binding("result", "result", result),),
        "workspaces": (),
        "semantic_attrs": {"axis": 1, "use_mean": False},
        "parameters": {"block_size": 128},
    }

    call = _norm_stats_call(raw)

    assert call["reduction_capacity"] == 60
    assert call["lane_count"] == 4
    assert "% 4" in call["source_offset"]
    assert "// 4" in call["source_offset"]
    assert len(call["result_offsets"]) == 1


def test_canonical_block_cyclic_source_uses_abi_coordinates_and_strides():
    source = _abi(shape=(2, 2, 3), strides=(37, 11, 2))
    source.update({
        "storage_kind": "canonical_global",
        "coordinate_space": "canonical_global",
        "logical_coordinate_expressions": (
            "local_coord_0",
            "local_coord_1 + shard_coord_0 * 2",
            "local_coord_2 + shard_coord_1 * 3",
        ),
        "distributed_type": {
            "placement": {"hierarchy": (2, 4)},
            "axis_policies": (
                {"kind": "broadcast"},
                {"kind": "split", "stages": ({"hierarchy_axes": (0,)},)},
                {"kind": "split", "stages": ({"hierarchy_axes": (1,)},)},
            ),
            "partial": None,
        },
    })
    result = _abi(shape=(1, 2, 1, 1), strides=(2, 1, 1, 1))
    raw = {
        "inputs": (_binding("input", "source", source),),
        "outputs": (_binding("result", "result", result),),
        "workspaces": (),
        "semantic_attrs": {"axis": 1, "use_mean": False},
        "parameters": {"block_size": 8},
    }

    call = _norm_stats_call(raw)

    assert "shard_y" in call["source_offset"]
    assert "shard_x" in call["source_offset"]
    assert "* 37" in call["source_offset"]
    assert "* 11" in call["source_offset"]
    assert "* 2" in call["source_offset"]
