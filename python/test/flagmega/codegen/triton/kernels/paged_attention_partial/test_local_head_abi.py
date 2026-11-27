# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.codegen.triton.kernel_call_renderers import (
    _paged_attention_partial_call,
)
from triton.flagmega.codegen.triton.templates import (
    KernelTemplateSpec,
    TritonTemplateRegistry,
)
from triton.flagmega.errors import CodegenError


def _abi(
    shape,
    *,
    lanes=1,
    dtype="float32",
    itemsize=4,
    storage="workspace",
    storage_kind="canonical_global",
    coordinate_space="canonical_global",
    owner_stride=0,
    partial=False,
    distributed=True,
):
    shape = tuple(shape)
    strides = []
    stride = 1
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return {
        "storage": storage,
        "storage_kind": storage_kind,
        "pool_byte_offset": 0,
        "scalar_dtype": dtype,
        "scalar_itemsize": itemsize,
        "logical_shape": shape,
        "local_capacity_shape": shape,
        "active_shape_expressions": tuple(str(value) for value in shape),
        "logical_coordinate_expressions": tuple(
            f"local_coord_{axis}" for axis in range(len(shape))
        ),
        "scalar_storage_strides": tuple(
            value * lanes for value in reversed(strides)
        ),
        "scalar_lane_shape": () if lanes == 1 else (lanes,),
        "scalar_lane_count": lanes,
        "component_stride_scalar_elements": owner_stride,
        "coordinate_space": coordinate_space,
        "distributed_type": (
            {
                "placement": {"hierarchy": (2, 1)},
                "axis_policies": tuple(
                    {"kind": "broadcast"} for _ in shape
                ),
                "partial": (
                    {"axes": (0,), "reduce_op": "sum"}
                    if partial
                    else None
                ),
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


def _parameter(formal, *bindings):
    return {"formal": formal, "buffers": bindings}


def _raw():
    query = _abi((1, 2, 2), lanes=4, dtype="bfloat16", itemsize=2)
    cache = _abi(
        (4, 1, 2, 16, 1, 2),
        lanes=4,
        dtype="bfloat16",
        itemsize=2,
        storage="input",
        storage_kind="compact_local",
        coordinate_space="local",
        distributed=False,
    )
    state = (
        cache,
        _abi((1,), dtype="int32", storage="input", distributed=False),
        _abi((1,), dtype="int32", storage="input", distributed=False),
        _abi((1,), dtype="int64", itemsize=8, storage="input", distributed=False),
        _abi((1, 4), dtype="int32", storage="input", distributed=False),
    )
    scalar = _abi((), dtype="int32", storage="scalar", distributed=False)
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
    return {
        "variant": "decode_t16",
        "semantic_attrs": {
            "layout": ("seq", "head", "dim"),
            "split_hierarchy_axis": 0,
            "split_count": 2,
            "scale": 0.5,
        },
        "parameters": {"token_tile": 4},
        "inputs": (
            _parameter("q", _binding("q", query)),
            _parameter(
                "state",
                *(_binding(f"state.{index}", abi) for index, abi in enumerate(state)),
            ),
            _parameter(
                "layer_id",
                _binding("layer_id", scalar, value_kind="immediate", argument="0"),
            ),
        ),
        "outputs": (
            _parameter("result_0", _binding("partial_max", stats)),
            _parameter("result_1", _binding("partial_sum", stats)),
            _parameter("result_2", _binding("partial_accumulator", accumulator)),
        ),
        "workspaces": (),
    }


def test_each_local_head_rebases_query_and_partial_state_pointers():
    call = _paged_attention_partial_call(_raw())

    assert call["head_dim"] == 8
    assert "attention_state_offsets" in call["query"]
    assert "* 8" in call["query"]
    assert "attention_state_offsets" in call["partial_max"]
    assert "attention_state_offsets" in call["partial_sum"]
    assert "* 8" in call["partial_accumulator"]


def test_partial_microkernel_indexes_only_within_rebased_head():
    implementation = KernelTemplateSpec(
        "paged_attention_partial", "decode_t16", "nvidia", "sm90"
    )
    source = TritonTemplateRegistry().render_kernel(implementation, {}).source

    compile(source, "paged_attention_partial.py", "exec")
    assert "query + query_offset" in source
    assert "(dimension // query_lanes) * query_dim_stride" in source
    assert "partial_accumulator + dimension * accumulator_dim_stride" in source
    assert "query + head * head_dim" not in source
    assert "partial_index = 0" in source


def test_query_and_cache_scalar_head_dimensions_must_agree():
    raw = _raw()
    cache = raw["inputs"][1]["buffers"][0]["abi"]
    cache["scalar_lane_count"] = 2
    cache["scalar_lane_shape"] = (2,)

    with pytest.raises(CodegenError, match="scalar head dimensions differ"):
        _paged_attention_partial_call(raw)
