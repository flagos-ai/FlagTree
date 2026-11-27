# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.call_abi import describe_local_buffer_abi
from triton.flagmega.codegen.triton.kernel_call_renderers import prepare_kernel_calls


def _descriptor(name, dtype, shape):
    nbytes = dtype.itemsize
    for extent in shape:
        nbytes *= extent
    physical = fm.PhysicalBuffer(
        f"physical:{name}", "workspace", fm.dim(nbytes), 16
    )
    strides = []
    stride = 1
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return fm.BufferDescriptor(
        name,
        dtype,
        shape,
        tuple(reversed(strides)),
        "workspace",
        16,
        fm.MemSpan(physical),
    )


def _binding(formal, descriptor):
    return {
        "formal": formal,
        "actual": descriptor.id,
        "runtime_argument": descriptor.id,
        "runtime_value_kind": "pointer",
        "abi": describe_local_buffer_abi(descriptor),
    }


def test_local_rms_norm_reduces_each_outer_row_and_expands_vector_lanes():
    vector = fm.vector_type("bfloat16", 4)
    value = _descriptor("value", vector, (2, 3, 2))
    weight = _descriptor("weight", vector, (2,))
    result = _descriptor("result", vector, (2, 3, 2))
    call = prepare_kernel_calls(
        ({
            "call": "norm",
            "family": "rms_norm",
            "variant": "local",
            "execution_kind": "local_shard",
            "parameters": {"block_size": 128},
            "semantic_attrs": {"epsilon": 1e-6, "weight_bias": 0.25},
            "inputs": (
                {"formal": "value", "buffers": (_binding("value", value),)},
                {"formal": "weight", "buffers": (_binding("weight", weight),)},
            ),
            "outputs": (
                {"formal": "result", "buffers": (_binding("result", result),)},
            ),
            "workspaces": (),
        },),
        function_name="main",
    )[0]

    assert call["outer_capacity"] == 6
    assert call["reduction_capacity"] == 8
    assert call["normalization_size"] == 8
    assert "% 4" in call["lane_coordinate"]
    assert "rms_reduction_offsets" in call["value_offset"]
    assert "rms_reduction_offsets" in call["weight_offset"]
    assert call["partial_stats"] is None
    assert call["writer_active"] == "True"
