# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from math import prod

import pytest

from triton.flagmega.codegen.triton.kernel_call_renderers import (
    prepare_kernel_calls,
)
from triton.flagmega.errors import CodegenError


def _abi(
    shape,
    *,
    dtype="bfloat16",
    itemsize=2,
    lanes=1,
    local_shape=None,
    coordinates=None,
    axis_policies=None,
    storage="input",
    storage_kind="canonical_global",
    coordinate_space="canonical_global",
    distributed=True,
):
    shape = tuple(shape)
    local_shape = tuple(shape if local_shape is None else local_shape)
    strides = []
    current = 1
    for extent in reversed(local_shape):
        strides.append(current)
        current *= extent
    strides = tuple(reversed(strides))
    return {
        "storage": storage,
        "storage_kind": storage_kind,
        "pool_byte_offset": 0,
        "scalar_dtype": dtype,
        "scalar_itemsize": itemsize,
        "logical_shape": shape,
        "local_capacity_shape": local_shape,
        "active_shape_expressions": tuple(str(value) for value in local_shape),
        "logical_coordinate_expressions": tuple(
            coordinates
            or (f"local_coord_{axis}" for axis in range(len(shape)))
        ),
        "scalar_storage_strides": tuple(value * lanes for value in strides),
        "scalar_lane_shape": () if lanes == 1 else (lanes,),
        "scalar_lane_count": lanes,
        "component_stride_scalar_elements": 0,
        "coordinate_space": coordinate_space,
        "distributed_type": (
            {
                "placement": {"hierarchy": (8, 16)},
                "axis_policies": tuple(
                    axis_policies
                    or ({"kind": "broadcast"} for _ in shape)
                ),
                "partial": None,
            }
            if distributed
            else None
        ),
    }


def _binding(name, abi, *, value_kind="pointer", argument=None):
    return {
        "formal": name,
        "actual": name,
        "runtime_argument": name if argument is None else argument,
        "runtime_value_kind": value_kind,
        "abi": abi,
    }


def _parameter(name, *bindings):
    return {"formal": name, "buffers": bindings}


def _raw(*, q_abi=None, q_result_abi=None, k_abi=None, v_abi=None):
    q_abi = q_abi or _abi((1, 2, 8))
    q_result_abi = q_result_abi or _abi((1, 2, 1), lanes=8)
    k_abi = k_abi or _abi((1, 1, 8))
    v_abi = v_abi or _abi((1, 1, 8))
    scale_abi = _abi((8,))
    frequency_abi = _abi((1, 1, 8), dtype="float32", itemsize=4)
    cache_abi = _abi(
        (16, 2, 2, 256, k_abi["logical_shape"][1], 1),
        lanes=8,
        storage_kind="compact_local",
        coordinate_space="local",
        distributed=False,
    )
    state_abis = (
        cache_abi,
        _abi((2,), dtype="int32", itemsize=4, distributed=False,
             storage_kind="compact_local", coordinate_space="local"),
        _abi((1,), dtype="int32", itemsize=4, distributed=False,
             storage_kind="compact_local", coordinate_space="local"),
        _abi((1,), dtype="int64", itemsize=8, distributed=False,
             storage_kind="compact_local", coordinate_space="local"),
        _abi((1, 16), dtype="int32", itemsize=4, distributed=False,
             storage_kind="compact_local", coordinate_space="local"),
    )
    scalar_abi = _abi(
        (), dtype="int32", itemsize=4, storage="scalar",
        storage_kind="compact_local", coordinate_space="local",
        distributed=False,
    )
    bool_abi = {**scalar_abi, "scalar_dtype": "bool", "scalar_itemsize": 1}
    state = tuple(
        _binding(f"state.{index}", abi) for index, abi in enumerate(state_abis)
    )
    return {
        "call": "qkv_rope_cache",
        "family": "qkv_rope_with_cache",
        "variant": "decode",
        "execution_kind": "local_shard",
        "parameters": {"elements_per_program": 128},
        "semantic_attrs": {
            "q_axis": -1,
            "q_epsilon": 1e-6,
            "q_use_mean": False,
            "k_axis": -1,
            "k_epsilon": 1e-6,
            "k_use_mean": False,
            "qkv_layout": ("seq", "head", "dim"),
            "attention_layout": ("seq", "head", "dim"),
        },
        "inputs": (
            _parameter(
                "qkv",
                _binding("q", q_abi),
                _binding("k", k_abi),
                _binding("v", v_abi),
            ),
            _parameter("q_scale", _binding("q_scale", scale_abi)),
            _parameter("k_scale", _binding("k_scale", scale_abi)),
            _parameter("q_bias", _binding("q_bias", scale_abi)),
            _parameter("k_bias", _binding("k_bias", scale_abi)),
            _parameter("cos", _binding("cos", frequency_abi)),
            _parameter("sin", _binding("sin", frequency_abi)),
            _parameter("state", *state),
            _parameter(
                "layer_id",
                _binding("layer_id", scalar_abi, value_kind="immediate", argument="3"),
            ),
            _parameter(
                "advance_sequence",
                _binding("advance", bool_abi, value_kind="immediate", argument="True"),
            ),
        ),
        "outputs": (
            _parameter("result_0", _binding("query_result", q_result_abi)),
            _parameter(
                "result_1",
                *(
                    _binding(f"state_result.{index}", abi)
                    for index, abi in enumerate(state_abis)
                ),
            ),
        ),
        "workspaces": (),
    }


def _prepare(**kwargs):
    return prepare_kernel_calls((_raw(**kwargs),), function_name="decode_layer")[0]


def test_decode_encoder_uses_only_local_buffer_abi_and_paged_state_contract():
    call = _prepare()

    assert call["family"] == "qkv_rope_with_cache"
    assert call["q"]["head_dim"] == 8
    assert call["q"]["normalization_size"] == 8
    assert call["k"]["head_dim"] == 8
    assert call["v"]["capacity"] == 8
    assert call["block_size"] == 256
    assert call["layer_id"] == "3"
    assert call["advance_sequence"] == "True"
    assert "qkv_cache_physical_block" in call["k"]["cache_offset"]
    assert "qkv_cache_physical_block" in call["v"]["cache_offset"]
    assert not any(
        name in repr(call).lower()
        for name in ("qwen", "sm90", "nvidia")
    )


def test_head_sharding_changes_coordinates_without_changing_kernel_family():
    split = (
        {"kind": "broadcast"},
        {"kind": "split", "stages": ({"hierarchy_axes": (1,)},)},
        {"kind": "broadcast"},
    )
    q_abi = _abi(
        (1, 16, 8),
        local_shape=(1, 1, 8),
        coordinates=(
            "local_coord_0",
            "local_coord_1 + shard_coord_1",
            "local_coord_2",
        ),
        axis_policies=split,
    )
    result_abi = _abi(
        (1, 16, 1),
        lanes=8,
        local_shape=(1, 1, 1),
        coordinates=(
            "local_coord_0",
            "local_coord_1 + shard_coord_1",
            "local_coord_2",
        ),
        axis_policies=split,
    )

    call = _prepare(q_abi=q_abi, q_result_abi=result_abi)

    assert call["family"] == "qkv_rope_with_cache"
    assert "shard_x" in call["q"]["output_offset"]
    assert call["q"]["writer_active"] == "(shard_y == 0)"


def test_k_and_v_use_distinct_redundant_writers_when_capacity_allows():
    kv_split = (
        {"kind": "broadcast"},
        {"kind": "split", "stages": ({"hierarchy_axes": (0,)},)},
        {"kind": "broadcast"},
    )
    coordinates = (
        "local_coord_0",
        "local_coord_1 + shard_y",
        "local_coord_2",
    )
    k_abi = _abi(
        (1, 8, 8),
        local_shape=(1, 1, 8),
        coordinates=coordinates,
        axis_policies=kv_split,
    )
    v_abi = _abi(
        (1, 8, 8),
        local_shape=(1, 1, 8),
        coordinates=coordinates,
        axis_policies=kv_split,
    )

    call = _prepare(k_abi=k_abi, v_abi=v_abi)

    assert call["k"]["writer_active"] == "(shard_x == 0)"
    assert call["v"]["writer_active"] == "(shard_x == 1)"


def test_normalized_dimension_sharding_requires_a_collective_variant():
    q_abi = _abi(
        (1, 2, 8),
        local_shape=(1, 2, 4),
        coordinates=(
            "local_coord_0",
            "local_coord_1",
            "local_coord_2 + shard_coord_1 * 4",
        ),
        axis_policies=(
            {"kind": "broadcast"},
            {"kind": "broadcast"},
            {"kind": "split", "stages": ({"hierarchy_axes": (1,)},)},
        ),
    )

    with pytest.raises(CodegenError, match="normalized suffix is sharded"):
        _prepare(q_abi=q_abi)
