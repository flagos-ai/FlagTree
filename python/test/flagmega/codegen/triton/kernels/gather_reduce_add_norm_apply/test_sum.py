# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.codegen.triton.kernel_call_renderers import (
    _gather_reduce_add_norm_apply_call,
)
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.errors import CodegenError


def _abi(
    shape,
    *,
    dtype="bfloat16",
    local_shape=None,
    storage_kind="canonical_global",
    coordinate_space="canonical_global",
    coordinates=None,
    axis_policies=None,
    partial_axes=None,
    owner_stride=0,
):
    local_shape = tuple(shape if local_shape is None else local_shape)
    raw_policies = tuple(
        axis_policies or ({"kind": "broadcast"} for _ in shape)
    )
    policies = tuple(
        {
            **policy,
            "stages": tuple(
                {
                    **stage,
                    "distribution": stage.get(
                        "distribution", {"kind": "contiguous"}
                    ),
                }
                for stage in policy.get("stages", ())
            ),
        }
        if policy.get("kind") == "split"
        else policy
        for policy in raw_policies
    )
    strides = []
    stride = 1
    for extent in reversed(local_shape):
        strides.append(stride)
        stride *= extent
    return {
        "storage": "input",
        "storage_kind": storage_kind,
        "pool_byte_offset": 0,
        "scalar_dtype": dtype,
        "scalar_itemsize": 4 if dtype == "float32" else 2,
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
            "kind": "distributed",
            "tensor": {
                "kind": "tensor",
                "dtype": dtype,
                "shape": tuple(
                    {"kind": "fixed", "value": value} for value in shape
                ),
                "layout": {},
            },
            "placement": {
                "hierarchy": (8, 16),
                "name": "yx",
                "hierarchy_levels": "bb",
            },
            "axis_policies": policies,
            "partial": (
                None
                if partial_axes is None
                else {
                    "kind": "partial",
                    "axes": tuple(partial_axes),
                    "reduce_op": "sum",
                }
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


def _raw(
    *,
    use_mean=False,
    partial_axes=(0, 1),
    partial_owner_count=128,
    partial=None,
    value=None,
    parameter=None,
):
    partial = partial or _abi(
        (1, 16),
        storage_kind="compact_per_owner",
        coordinate_space="local",
        partial_axes=partial_axes,
        owner_stride=16,
    )
    value = value or _abi((1, 16))
    parameter = parameter or _abi((16,))
    components = 2 if use_mean else 1
    workspace = _abi((components, 128), dtype="float32")
    return {
        "semantic_attrs": {
            "axis": -1,
            "epsilon": 1e-6,
            "use_mean": use_mean,
            "has_bias": True,
        },
        "parameters": {
            "tile": 16,
            "partial_reduction_width": 32,
            "reduction_width": 128,
            "owner_count": 128,
            "partial_owner_count": partial_owner_count,
            "partial_axes": partial_axes,
        },
        "inputs": (
            _parameter("input", partial),
            _parameter("addend", value),
            _parameter("scale", parameter),
            _parameter("bias", parameter),
        ),
        "outputs": (
            _parameter("result_0", value),
            _parameter("result_1", value),
        ),
        "workspaces": (
            _parameter("norm_stats_partials", workspace),
        ),
    }


@pytest.mark.parametrize("use_mean", [False, True])
def test_renderer_materializes_private_stats_with_one_grid_barrier(
    use_mean, assert_entry_owned_phase_schedule,
):
    call = {
        **_gather_reduce_add_norm_apply_call(_raw(use_mean=use_mean)),
        "symbol": "_flagmega_test_gather_reduce_add_norm_apply",
        "signature": (
            "input, addend, scale, bias, result_0, result_1, "
            "norm_stats_partials"
        ),
        "execution_kind": "collective",
        "family": "gather_reduce_add_norm_apply",
        "variant": "sum",
        "internal_grid_barriers": 1,
    }
    source = TritonTemplateRegistry().render(
        "kernels/gather_reduce_add_norm_apply/sum.py.jinja",
        {
            "render_calls": (call,),
            "distributed_entry": True,
            "mesh_axis_names": ("y", "x"),
            "mesh_hierarchy": (8, 16),
        },
    )

    compile(source, "gather_reduce_add_norm_apply.py", "exec")
    assert_entry_owned_phase_schedule(source, call["symbol"])
    annotation = source.rsplit(
        "def _flagmega_test_gather_reduce_add_norm_apply", 1
    )[0].rstrip().splitlines()[-1]
    assert annotation == "@triton.jit"
    assert source.count("tle.distributed_barrier(FLAGMEGA_GRID_MESH)") == 1
    assert "norm_owner_start in tl.range" in source
    assert "gather_partial_lane = gather_partial_start + tl.arange" in source
    assert "gather_partial_owner[None, :]" in source
    assert "axis=1" in source
    assert "for gather_partial_member in tl.range" not in source
    assert "tl.sum(gather_square_sum, axis=0)" in source
    assert call["stats_writer_index"] == "((shard_y) * 16 + (shard_x))"
    assert call["work_partition_count"] == 128
    assert (
        f"+ {128 if use_mean else 0} + {call['stats_writer_index']}" in source
    )
    if use_mean:
        assert "tl.sum(gather_mean_sum, axis=0)" in source
    assert "collective" not in call


def test_split_y_partial_x_partitions_materialization_across_all_writers():
    split_y = (
        {"kind": "broadcast"},
        {"kind": "split", "stages": ({"hierarchy_axes": (0,)},)},
    )
    partial = _abi(
        (1, 2048),
        local_shape=(1, 256),
        storage_kind="compact_per_owner",
        coordinate_space="local",
        coordinates=("local_coord_0", "local_coord_1 + shard_coord_0 * 256"),
        axis_policies=split_y,
        partial_axes=(1,),
        owner_stride=256,
    )
    value = _abi(
        (1, 2048),
        local_shape=(1, 256),
        coordinates=("local_coord_0", "local_coord_1 + shard_coord_0 * 256"),
        axis_policies=split_y,
    )
    parameter = _abi(
        (2048,),
        local_shape=(256,),
        coordinates=("local_coord_0 + shard_coord_0 * 256",),
        axis_policies=(split_y[1],),
    )

    call = _gather_reduce_add_norm_apply_call(
        _raw(
            partial_axes=(1,),
            partial_owner_count=16,
            partial=partial,
            value=value,
            parameter=parameter,
        )
    )

    assert call["partial_reduction_width"] == 16
    assert call["partial_owner"] == (
        "(shard_y * 16 + (gather_partial_lane))"
    )
    assert call["participant_active"] == "True"
    assert call["stats_writer_index"] == "((shard_y) * 16 + (shard_x))"
    assert call["stats_owner_count"] == 128
    assert call["work_partition_index"] == "(shard_x)"
    assert call["work_partition_count"] == 16
    assert call["tile"] == 16


def test_down_projection_split_yx_uses_all_128_stats_writers():
    split_yx = (
        {"kind": "broadcast"},
        {
            "kind": "split",
            "stages": ({"hierarchy_axes": (0, 1)},),
        },
    )
    partial = _abi(
        (1, 2048),
        local_shape=(1, 2048),
        storage_kind="compact_per_owner",
        coordinate_space="local",
        partial_axes=(0, 1),
        owner_stride=2048,
    )
    value = _abi(
        (1, 2048),
        local_shape=(1, 16),
        coordinates=(
            "local_coord_0",
            "local_coord_1 + (shard_coord_0 * 16 + shard_coord_1) * 16",
        ),
        axis_policies=split_yx,
    )
    parameter = _abi(
        (2048,),
        local_shape=(16,),
        coordinates=(
            "local_coord_0 + (shard_coord_0 * 16 + shard_coord_1) * 16",
        ),
        axis_policies=(split_yx[1],),
    )

    call = _gather_reduce_add_norm_apply_call(
        _raw(partial=partial, value=value, parameter=parameter)
    )

    assert call["partial_reduction_width"] == 32
    assert call["partial_owner"] == "(gather_partial_lane)"
    assert call["participant_active"] == "True"
    assert call["stats_writer_index"] == "((shard_y) * 16 + (shard_x))"
    assert call["stats_owner_count"] == 128
    assert call["work_partition_count"] == 1


def test_down_projection_split_y_partial_x_uses_all_output_shards():
    split_y = (
        {"kind": "broadcast"},
        {"kind": "split", "stages": ({"hierarchy_axes": (0,)},)},
    )
    split_yx = (
        {"kind": "broadcast"},
        {"kind": "split", "stages": ({"hierarchy_axes": (0, 1)},)},
    )
    partial = _abi(
        (1, 2048),
        local_shape=(1, 256),
        storage_kind="compact_per_owner",
        coordinate_space="local",
        coordinates=("local_coord_0", "local_coord_1 + shard_coord_0 * 256"),
        axis_policies=split_y,
        partial_axes=(1,),
        owner_stride=256,
    )
    value = _abi(
        (1, 2048),
        local_shape=(1, 16),
        coordinates=(
            "local_coord_0",
            "local_coord_1 + (shard_coord_0 * 16 + shard_coord_1) * 16",
        ),
        axis_policies=split_yx,
    )
    parameter = _abi(
        (2048,),
        local_shape=(16,),
        coordinates=(
            "local_coord_0 + (shard_coord_0 * 16 + shard_coord_1) * 16",
        ),
        axis_policies=(split_yx[1],),
    )

    call = _gather_reduce_add_norm_apply_call(
        _raw(
            partial_axes=(1,),
            partial_owner_count=16,
            partial=partial,
            value=value,
            parameter=parameter,
        )
    )

    assert call["local_capacity"] == 16
    assert call["participant_active"] == "True"
    assert call["stats_writer_index"] == "((shard_y) * 16 + (shard_x))"
    assert call["stats_owner_count"] == 128
    assert "shard_x" in call["value_offset"]
    assert "shard_x" in call["scale_offset"]


def test_renderer_rejects_owner_count_that_disagrees_with_placement():
    raw = _raw()
    raw["parameters"]["owner_count"] = 64

    with pytest.raises(CodegenError, match="owner counts disagree"):
        _gather_reduce_add_norm_apply_call(raw)


def test_renderer_rejects_materialized_partial_storage():
    raw = _raw()
    raw["inputs"][0]["buffers"][0]["abi"]["storage_kind"] = "canonical_global"

    with pytest.raises(CodegenError, match="compact per-owner"):
        _gather_reduce_add_norm_apply_call(raw)
