# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.kernel_call_renderers import (
    _paged_attention_combine_call,
)
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry


def _abi(
    shape,
    *,
    local_shape=None,
    logical_coordinates=None,
    lanes=1,
    dtype="float32",
    itemsize=4,
    storage_kind="canonical_global",
    coordinate_space="canonical_global",
    owner_stride=0,
    partial=False,
):
    shape = tuple(shape)
    local_shape = shape if local_shape is None else tuple(local_shape)
    strides = []
    stride = 1
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return {
        "storage": "workspace",
        "storage_kind": storage_kind,
        "pool_byte_offset": 0,
        "scalar_dtype": dtype,
        "scalar_itemsize": itemsize,
        "logical_shape": shape,
        "local_capacity_shape": local_shape,
        "active_shape_expressions": tuple(str(value) for value in local_shape),
        "logical_coordinate_expressions": (
            tuple(logical_coordinates)
            if logical_coordinates is not None
            else tuple(f"local_coord_{axis}" for axis in range(len(shape)))
        ),
        "scalar_storage_strides": tuple(
            value * lanes for value in reversed(strides)
        ),
        "scalar_lane_shape": () if lanes == 1 else (lanes,),
        "scalar_lane_count": lanes,
        "component_stride_scalar_elements": owner_stride,
        "coordinate_space": coordinate_space,
        "distributed_type": {
            "placement": {"hierarchy": (2, 1)},
            "axis_policies": tuple({"kind": "broadcast"} for _ in shape),
            "partial": (
                {"axes": (0,), "reduce_op": "sum"} if partial else None
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


def _raw():
    stats = _abi(
        (1, 2, 1),
        storage_kind="compact_per_owner",
        coordinate_space="local",
        owner_stride=2,
        partial=True,
    )
    accumulator = _abi(
        (1, 2, 8),
        storage_kind="compact_per_owner",
        coordinate_space="local",
        owner_stride=16,
        partial=True,
    )
    result = _abi(
        (1, 2, 2), lanes=4, dtype="bfloat16", itemsize=2
    )
    return {
        "semantic_attrs": {
            "layout": ("seq", "head", "dim"),
            "split_hierarchy_axis": 0,
            "split_count": 2,
        },
        "parameters": {"elements_per_program": 8},
        "inputs": (
            _parameter("max_state", stats),
            _parameter("sum_state", stats),
            _parameter("acc_state", accumulator),
        ),
        "outputs": (_parameter("result", result),),
        "workspaces": (),
    }


def test_vector_result_domain_visits_every_scalar_lane():
    call = _paged_attention_combine_call(_raw())

    assert call["local_capacity"] == 16
    assert call["head_dim"] == 8
    assert "% 4" in call["dimension"]
    assert "% 4" in call["result_offset"]
    assert "% 4" in call["partial_accumulator_offset"]
    assert "attention_output_offsets" in call["partial_accumulator_offset"]


def test_rendered_vector_result_loop_covers_scalar_head_dimension():
    call = {
        **_paged_attention_combine_call(_raw()),
        "symbol": "_flagmega_test_attention_combine",
        "signature": "max_state, sum_state, acc_state, result",
        "execution_kind": "collective",
        "family": "paged_attention_combine",
        "variant": "decode",
        "internal_grid_barriers": 0,
    }
    source = TritonTemplateRegistry().render(
        "kernels/paged_attention_combine/decode.py.jinja",
        {
            "render_calls": (call,),
            "distributed_entry": True,
            "mesh_axis_names": ("y", "x"),
            "mesh_hierarchy": (2, 1),
        },
    )

    compile(source, "paged_attention_combine_call_graph.py", "exec")
    assert "tl.range(0, 16, 8)" in source
    assert "% 4" in source
    annotation = source.rsplit(
        "def _flagmega_test_attention_combine", 1
    )[0].rstrip().splitlines()[-1]
    assert annotation == "@triton.jit(noinline=True)"


def test_local_shard_wrapper_keeps_op_boundary_without_entry_synchronization():
    call = {
        **_paged_attention_combine_call(_raw()),
        "symbol": "_flagmega_test_local_wrapper",
        "signature": "max_state, sum_state, acc_state, result",
        "execution_kind": "local_shard",
        "family": "paged_attention_combine",
        "variant": "decode",
        "internal_grid_barriers": 0,
    }
    source = TritonTemplateRegistry().render(
        "kernels/paged_attention_combine/decode.py.jinja",
        {
            "render_calls": (call,),
            "distributed_entry": True,
            "mesh_axis_names": ("y", "x"),
            "mesh_hierarchy": (2, 1),
        },
    )

    annotation = source.rsplit(
        "def _flagmega_test_local_wrapper", 1
    )[0].rstrip().splitlines()[-1]
    assert annotation == "@triton.jit(noinline=True)"


def test_dim_sharded_output_uses_global_dimension_for_broadcast_partial_state():
    raw = _raw()
    result = _abi(
        (1, 2, 2),
        local_shape=(1, 1, 1),
        logical_coordinates=(
            "local_coord_0",
            "shard_coord_1 + local_coord_1",
            "shard_coord_0 + local_coord_2",
        ),
        lanes=4,
        dtype="bfloat16",
        itemsize=2,
    )
    raw["outputs"] = (_parameter("result", result),)

    call = _paged_attention_combine_call(raw)

    assert "shard_y" in call["dimension"]
    assert "shard_y" in call["partial_accumulator_offset"]
