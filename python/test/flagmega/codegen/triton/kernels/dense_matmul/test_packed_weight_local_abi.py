# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.kernel_call_renderers import (
    _dense_matmul_call,
    _dense_matmul_glu_call,
)


def _abi(
    *,
    logical_shape,
    local_shape,
    scalar_strides,
    lane_shape=(),
    storage_kind="compact_local",
    coordinate_space="local",
    logical_coordinates=None,
    component_stride=0,
    pool_byte_offset=0,
):
    lane_count = 1
    for extent in lane_shape:
        lane_count *= extent
    if logical_coordinates is None:
        logical_coordinates = tuple(
            f"local_coord_{axis}" for axis in range(len(logical_shape))
        )
    return {
        "storage": "workspace",
        "pool_byte_offset": pool_byte_offset,
        "storage_kind": storage_kind,
        "scalar_dtype": "bfloat16",
        "scalar_itemsize": 2,
        "logical_shape": tuple(logical_shape),
        "local_capacity_shape": tuple(local_shape),
        "active_shape_expressions": tuple(str(value) for value in local_shape),
        "logical_coordinate_expressions": tuple(logical_coordinates),
        "scalar_storage_strides": tuple(scalar_strides),
        "scalar_lane_shape": tuple(lane_shape),
        "scalar_lane_count": lane_count,
        "component_stride_elements": component_stride // lane_count,
        "component_stride_scalar_elements": component_stride,
        "coordinate_space": coordinate_space,
    }


def _binding(abi):
    return {"abi": abi, "runtime_argument": "workspace"}


def _parameter(formal, abi):
    return {"formal": formal, "buffers": [_binding(abi)]}


def _raw_call(weight_abi, *, packed_layout="k_major_n8_k16"):
    source_abi = _abi(
        logical_shape=(1, 32),
        local_shape=(1, 32),
        scalar_strides=(32, 1),
    )
    result_abi = _abi(
        logical_shape=(1, 16),
        local_shape=(1, 8),
        scalar_strides=(8, 1),
        storage_kind="compact_per_owner",
        logical_coordinates=("local_coord_0", "local_coord_1 + shard_coord_1 * 8"),
        component_stride=8,
    )
    return {
        "semantic_op": "ntt.packed_matmul",
        "variant": "packed_k_major_gemv" if packed_layout else "gemv",
        "semantic_attrs": {"transpose_b": True},
        "parameters": {
            "packed_layout": packed_layout,
            "block_k": 32,
            "tile_n": 8,
        },
        "inputs": [
            _parameter("lhs", source_abi),
            _parameter("rhs", weight_abi),
        ],
        "outputs": [_parameter("result", result_abi)],
    }


def _typed_weight_abi(*, compact_per_owner):
    return _abi(
        logical_shape=(2, 2),
        local_shape=(2, 1) if compact_per_owner else (2, 2),
        scalar_strides=(128, 128) if compact_per_owner else (256, 128),
        lane_shape=(8, 2, 8),
        storage_kind=(
            "compact_per_owner" if compact_per_owner else "canonical_global"
        ),
        coordinate_space="local" if compact_per_owner else "canonical_global",
        logical_coordinates=(
            "local_coord_0",
            "local_coord_1 + shard_coord_1" if compact_per_owner else "local_coord_1",
        ),
        component_stride=256 if compact_per_owner else 0,
        pool_byte_offset=2048,
    )


def test_compact_owner_packed_weight_uses_owner_base_and_local_coordinates():
    call = _dense_matmul_call(
        _raw_call(_typed_weight_abi(compact_per_owner=True))
    )

    assert "(shard_index) * 256" in call["weight"]
    assert "dense_local_n_offsets" in call["weight_offset"]
    assert "dense_local_k_offsets" in call["weight_offset"]
    assert "dense_global_n" not in call["weight_offset"]
    assert "dense_global_k" not in call["weight_offset"]


def test_canonical_packed_weight_uses_storage_base_and_global_coordinates():
    call = _dense_matmul_call(
        _raw_call(_typed_weight_abi(compact_per_owner=False))
    )

    assert "shard_index" not in call["weight"]
    assert "dense_global_n" in call["weight_offset"]
    assert "dense_global_k" in call["weight_offset"]


def test_legacy_scalar_physical_packing_uses_the_same_local_abi_rule():
    weight_abi = _abi(
        logical_shape=(2, 2, 2, 64),
        local_shape=(2, 1, 2, 64),
        scalar_strides=(128, 128, 64, 1),
        storage_kind="compact_per_owner",
        logical_coordinates=(
            "local_coord_0",
            "local_coord_1 + shard_coord_1",
            "local_coord_2",
            "local_coord_3",
        ),
        component_stride=256,
    )
    call = _dense_matmul_call(_raw_call(weight_abi))

    assert "(shard_index) * 256" in call["weight"]
    assert "dense_local_n_offsets" in call["weight_offset"]
    assert "dense_local_k_offsets" in call["weight_offset"]
    assert "// 64" in call["weight_offset"]


def test_unpacked_compact_rhs_uses_local_dense_coordinates():
    weight_abi = _abi(
        logical_shape=(16, 32),
        local_shape=(8, 32),
        scalar_strides=(32, 1),
        storage_kind="compact_per_owner",
        logical_coordinates=(
            "local_coord_0 + shard_coord_1 * 8",
            "local_coord_1",
        ),
        component_stride=256,
    )
    call = _dense_matmul_call(_raw_call(weight_abi, packed_layout=None))

    assert "(shard_index) * 256" in call["weight"]
    assert "dense_local_n_offsets" in call["weight_offset"]
    assert "dense_local_k_offsets" in call["weight_offset"]
    assert "dense_global_n" not in call["weight_offset"]


def test_glu_keeps_independent_storage_coordinates_for_each_rhs():
    compact = _typed_weight_abi(compact_per_owner=True)
    canonical = _typed_weight_abi(compact_per_owner=False)
    source_abi = _abi(
        logical_shape=(1, 32),
        local_shape=(1, 32),
        scalar_strides=(32, 1),
    )
    result_abi = _abi(
        logical_shape=(1, 16),
        local_shape=(1, 8),
        scalar_strides=(8, 1),
        storage_kind="compact_per_owner",
        logical_coordinates=("local_coord_0", "local_coord_1 + shard_coord_1 * 8"),
        component_stride=8,
    )
    raw = {
        "variant": "packed_k_major_gemv",
        "parameters": {
            "packed_layout": "k_major_n8_k16",
            "block_k": 32,
            "tile_n": 8,
        },
        "inputs": [
            _parameter("value", source_abi),
            _parameter("gate_weight", compact),
            _parameter("up_weight", canonical),
        ],
        "outputs": [_parameter("result", result_abi)],
    }

    call = _dense_matmul_glu_call(raw)

    assert "shard_index" in call["gate_weight"]
    assert "dense_glu_local_n_offsets" in call["gate_weight_offset"]
    assert "shard_index" not in call["up_weight"]
    assert "dense_glu_global_n" in call["up_weight_offset"]
