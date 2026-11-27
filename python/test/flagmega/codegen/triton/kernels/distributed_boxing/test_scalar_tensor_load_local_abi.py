# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.kernel_call_renderers import (
    prepare_kernel_calls,
)


def _scalar_abi():
    return {
        "storage": "scalar",
        "storage_kind": "compact_local",
        "pool_byte_offset": 0,
        "scalar_dtype": "int32",
        "scalar_itemsize": 4,
        "logical_shape": (),
        "local_capacity_shape": (),
        "active_shape_expressions": (),
        "logical_coordinate_expressions": (),
        "scalar_storage_strides": (),
        "scalar_lane_shape": (),
        "scalar_lane_count": 1,
        "component_stride_scalar_elements": 0,
        "coordinate_space": "local",
        "distributed_type": None,
    }


def _distributed_scalar_abi():
    return {
        **_scalar_abi(),
        "storage": "workspace",
        "storage_kind": "canonical_global",
        "coordinate_space": "canonical_global",
        "distributed_type": {
            "placement": {"hierarchy": (8, 16)},
            "axis_policies": (),
            "partial": None,
        },
    }


def test_tensor_load_uses_immediate_value_instead_of_forming_a_pointer():
    raw = {
        "call": "load_layer_id",
        "family": "distributed_boxing",
        "variant": "tensor_load",
        "execution_kind": "collective",
        "parameters": {"tile": 1, "leaf_transitions": ("tensor_load",)},
        "semantic_attrs": {},
        "inputs": ({
            "formal": "value",
            "buffers": ({
                "formal": "value",
                "actual": "layer_id",
                "runtime_argument": "7",
                "runtime_value_kind": "immediate",
                "abi": _scalar_abi(),
            },),
        },),
        "outputs": ({
            "formal": "result",
            "buffers": ({
                "formal": "result",
                "actual": "layer_id.distributed",
                "runtime_argument": "workspace",
                "runtime_value_kind": "pointer",
                "abi": _distributed_scalar_abi(),
            },),
        },),
        "workspaces": (),
    }

    call = prepare_kernel_calls((raw,), function_name="main")[0]
    leaf = call["leaves"][0]

    assert call["arguments"] == ["workspace"]
    assert leaf["source_is_scalar"] is True
    assert leaf["source"] == "7"
    assert leaf["result"] == "workspace.to(tl.pointer_type(tl.int32))"
    assert leaf["capacity"] == 1
