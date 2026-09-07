# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.codegen.triton.kernel_call_renderers import (
    _gather_reduce_add_norm_stats_call,
)
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
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
):
    local_shape = tuple(shape if local_shape is None else local_shape)
    strides = []
    stride = 1
    for extent in reversed(local_shape):
        strides.append(stride)
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
        "scalar_storage_strides": tuple(reversed(strides)),
        "scalar_lane_shape": (),
        "scalar_lane_count": 1,
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


def _raw(partial_abi, *, partial_axes, partial_owner_count, stats_abi=None):
    value_abi = _abi((1, 2048))
    if stats_abi is None:
        stats_abi = _abi((1, 1, 1))
        stats_abi["scalar_dtype"] = "float32"
        stats_abi["scalar_itemsize"] = 4
    return {
        "semantic_attrs": {"axis": -1, "use_mean": False},
        "parameters": {
            "tile": 16,
            "partial_reduction_width": 16,
            "owner_count": 128,
            "partial_owner_count": partial_owner_count,
            "partial_axes": partial_axes,
        },
        "inputs": (
            _parameter("input", partial_abi),
            _parameter("addend", value_abi),
        ),
        "outputs": (
            _parameter("result_0", value_abi),
            _parameter("result_1", stats_abi),
        ),
        "workspaces": (
            _parameter("collective", value_abi),
            _parameter("norm_stats_partials", _abi((128,))),
        ),
    }


def test_hybrid_partial_group_reads_owner_components_in_one_local_shard():
    partial = _abi(
        (1, 2048),
        local_shape=(1, 256),
        storage_kind="compact_per_owner",
        coordinate_space="local",
        coordinates=("local_coord_0", "local_coord_1 + shard_coord_0 * 256"),
        axis_policies=(
            {"kind": "broadcast"},
            {"kind": "split", "stages": ({"hierarchy_axes": (0,)},)},
        ),
        partial_axes=(1,),
        owner_stride=256,
    )

    stats = _abi(
        (1, 1, 1),
        storage_kind="compact_per_owner",
        partial_axes=(0,),
        owner_stride=1,
    )
    stats["scalar_dtype"] = "float32"
    stats["scalar_itemsize"] = 4
    call = _gather_reduce_add_norm_stats_call(
        _raw(
            partial,
            partial_axes=(1,),
            partial_owner_count=16,
            stats_abi=stats,
        )
    )

    assert call["local_capacity"] == 256
    assert call["owner_stride"] == 256
    assert call["participant_active"] == "True"
    assert call["work_partition_index"] == "(shard_x)"
    assert call["work_partition_count"] == 16
    assert call["partial_owner"] == "(shard_y * 16 + (gather_partial_lane))"
    assert "shard_y" in call["residual_offset"]
    assert "shard_y" in call["result_offset"]


def test_hybrid_partial_statistics_keep_y_partials_broadcast_across_x():
    partial = _abi(
        (1, 2048),
        local_shape=(1, 256),
        storage_kind="compact_per_owner",
        coordinate_space="local",
        coordinates=("local_coord_0", "local_coord_1 + shard_coord_0 * 256"),
        axis_policies=(
            {"kind": "broadcast"},
            {"kind": "split", "stages": ({"hierarchy_axes": (0,)},)},
        ),
        partial_axes=(1,),
        owner_stride=256,
    )

    stats = _abi(
        (1, 1, 1),
        storage_kind="compact_per_owner",
        partial_axes=(0,),
        owner_stride=1,
    )
    stats["scalar_dtype"] = "float32"
    stats["scalar_itemsize"] = 4
    call = _gather_reduce_add_norm_stats_call(
        _raw(
            partial,
            partial_axes=(1,),
            partial_owner_count=16,
            stats_abi=stats,
        )
    )

    # P(y) is broadcast over x: each x replica must read the representative
    # x=0 partial for its y group, rather than consuming one global total.
    assert call["stats_source_owner"] == "(shard_y * 16 + (0))"
    assert call["stats_partial_owner"] == (
        "(shard_y * 16 + (gather_stats_lane))"
    )
    assert call["stats_reduction_count"] == 16
    assert call["stats_store_active"] == "True"


def test_all_axis_partial_group_has_one_representative_and_full_owner_range():
    partial = _abi(
        (1, 2048),
        storage_kind="compact_per_owner",
        coordinate_space="local",
        partial_axes=(0, 1),
        owner_stride=2048,
    )

    call = _gather_reduce_add_norm_stats_call(
        _raw(partial, partial_axes=(0, 1), partial_owner_count=128)
    )

    assert call["participant_active"] == "True"
    assert call["work_partition_index"] == "(shard_index)"
    assert call["work_partition_count"] == 128
    assert call["partial_owner"] == "(gather_partial_lane)"
    assert call["partial_owner_count"] == 128
    assert call["stats_partial_owner"] == "(gather_stats_lane)"
    assert call["stats_reduction_count"] == 128


def test_partial_value_without_per_owner_storage_is_rejected():
    partial = _abi((1, 2048), partial_axes=(0, 1))

    with pytest.raises(CodegenError, match="compact component per placement owner"):
        _gather_reduce_add_norm_stats_call(
            _raw(partial, partial_axes=(0, 1), partial_owner_count=128)
        )


def test_owner_reduction_loads_bounded_owner_vectors_instead_of_scalar_gathers(
    assert_entry_owned_phase_schedule,
):
    partial = _abi(
        (1, 2048),
        storage_kind="compact_per_owner",
        coordinate_space="local",
        partial_axes=(0, 1),
        owner_stride=2048,
    )
    call = {
        **_gather_reduce_add_norm_stats_call(
            _raw(partial, partial_axes=(0, 1), partial_owner_count=128)
        ),
        "symbol": "_flagmega_test_gather_norm",
        "signature": (
            "input, addend, result_0, result_1, collective, "
            "norm_stats_partials"
        ),
        "execution_kind": "collective",
        "family": "gather_reduce_add_norm_stats",
        "variant": "sum_rms",
        "internal_grid_barriers": 1,
    }
    source = TritonTemplateRegistry().render(
        "kernels/gather_reduce_add_norm_stats/sum_rms.py.jinja",
        {
            "render_calls": (call,),
            "distributed_entry": True,
            "mesh_axis_names": ("y", "x"),
            "mesh_hierarchy": (8, 16),
        },
    )

    compile(source, "gather_norm_call_graph.py", "exec")
    assert_entry_owned_phase_schedule(source, call["symbol"])
    assert call["partial_reduction_width"] == 16
    assert "gather_partial_lane = gather_partial_start + tl.arange" in source
    assert "gather_partial_owner[None, :]" in source
    assert "gather_mask[:, None]" in source
    assert "axis=1" in source
    assert "for gather_partial_member in tl.range" not in source
    assert "(shard_index) * 16" in source
    assert "128 * 16" in source
    assert "gather_stats_lane = gather_stats_start + tl.arange" in source
    assert "tl.sum(" in source
    assert "tl.static_range" not in source
