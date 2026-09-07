# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Unit coverage for distributed tensor-map table geometry."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.tensor_descriptor_planner import (
    packed_distributed_tensor_map_table_request,
    packed_owner_prefix_tensor_map_table_request,
)
from triton.flagmega.errors import CodegenError


def _down_weight_abi():
    distributed = fm.DistributedType(
        fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (384, 256)),
        (
            fm.SBP.split_contiguous((1,), 24),
            fm.SBP.split_contiguous((0,), 32),
        ),
        fm.Placement((8, 16), "yx", "bb"),
    )
    return {
        "coordinate_space": "canonical_global",
        "distributed_type": distributed.to_data(),
        "logical_shape": (384, 256),
        "local_capacity_shape": (24, 32),
        "scalar_dtype": "bfloat16",
    }


def _request(abi):
    return packed_distributed_tensor_map_table_request(
        abi,
        parameter="weight_descriptor",
        source="rdata",
        offset_bytes=4096,
        descriptor_shape=(384, 256, 2, 64),
        descriptor_strides=(32768, 128, 64, 1),
        block_shape=(8, 8, 2, 64),
    )


def test_table_rebases_disjoint_k_and_n_splits_in_mesh_linear_order():
    request = _request(_down_weight_abi())
    entries = request["entries"]

    assert request["kind"] == "table"
    assert request["swizzle_mode"] == 3
    assert len(entries) == 128
    assert entries[0]["shape"] == (24, 32, 2, 64)
    assert entries[0]["offset_bytes"] == 4096
    # Row-major owner order is (y, x), so owner one advances K and owner
    # sixteen advances N.
    assert entries[1]["offset_bytes"] == 4096 + 24 * 32768 * 2
    assert entries[16]["offset_bytes"] == 4096 + 32 * 128 * 2
    assert entries[-1]["offset_bytes"] == (
        4096 + (15 * 24 * 32768 + 7 * 32 * 128) * 2
    )


def test_table_preserves_ragged_owner_extent_for_tma_zero_padding():
    abi = {
        "coordinate_space": "canonical_global",
        "distributed_type": {
            "kind": "distributed",
            "placement": {
                "hierarchy": [3],
                "name": "x",
                "hierarchy_levels": "b",
            },
            "axis_policies": [{
                "kind": "split",
                "stages": [{
                    "hierarchy_axes": [0],
                    "distribution": {
                        "kind": "block_cyclic",
                        "block_size": 4,
                    },
                }],
            }],
        },
        "logical_shape": (10,),
        "local_capacity_shape": (4,),
        "scalar_dtype": "bfloat16",
    }
    request = packed_distributed_tensor_map_table_request(
        abi,
        parameter="weight_descriptor",
        source="rdata",
        offset_bytes=0,
        descriptor_shape=(10, 2, 64),
        descriptor_strides=(128, 64, 1),
        block_shape=(4, 2, 64),
    )

    assert tuple(entry["shape"][0] for entry in request["entries"]) == (4, 4, 2)
    assert tuple(entry["offset_bytes"] for entry in request["entries"]) == (
        0,
        4 * 128 * 2,
        8 * 128 * 2,
    )


def test_table_rejects_non_rectangular_block_cyclic_policy():
    abi = _down_weight_abi()
    abi["distributed_type"] = {
        **abi["distributed_type"],
        "axis_policies": [
            {
                "kind": "split",
                "stages": [{
                    "hierarchy_axes": [1],
                    "distribution": {"kind": "block_cyclic", "block_size": 2},
                }],
            },
            abi["distributed_type"]["axis_policies"][1],
        ],
    }

    with pytest.raises(CodegenError, match="not an affine shard"):
        _request(abi)


def test_table_encodes_block_cyclic_n_as_affine_strided_local_coordinates():
    distributed = fm.DistributedType(
        fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (384, 256)),
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0, 1), 1)),
        fm.Placement((8, 16), "yx", "bb"),
    )
    abi = {**_down_weight_abi(), "distributed_type": distributed.to_data(), "local_capacity_shape": (384, 2)}
    entries = _request(abi)["entries"]
    for owner, entry in enumerate(entries):
        assert entry["shape"] == (384, 2, 2, 64)
        assert entry["strides"] == (32768, 16384, 64, 1)
        assert entry["offset_bytes"] == 4096 + owner * 128 * 2


def test_owner_prefix_table_rebases_a_dense_physical_owner_axis():
    request = packed_owner_prefix_tensor_map_table_request(
        {"scalar_dtype": "bfloat16"},
        parameter="weight_descriptor",
        source="rdata",
        offset_bytes=4096,
        owner_count=128,
        descriptor_shape=(128, 32, 16, 2, 64),
        descriptor_strides=(65536, 128, 4096, 64, 1),
        block_shape=(1, 32, 4, 2, 64),
    )

    assert request["block_shape"] == (32, 4, 2, 64)
    assert len(request["entries"]) == 128
    assert request["entries"][0]["shape"] == (32, 16, 2, 64)
    assert request["entries"][0]["strides"] == (128, 4096, 64, 1)
    assert request["entries"][1]["offset_bytes"] == 4096 + 65536 * 2


def test_owner_prefix_table_rejects_overlapping_owner_payloads():
    with pytest.raises(CodegenError, match="overlap"):
        packed_owner_prefix_tensor_map_table_request(
            {"scalar_dtype": "bfloat16"},
            parameter="weight_descriptor",
            source="rdata",
            offset_bytes=0,
            owner_count=2,
            descriptor_shape=(2, 8, 8),
            descriptor_strides=(32, 8, 1),
            block_shape=(1, 8, 8),
        )
