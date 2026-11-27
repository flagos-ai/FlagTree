# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.call_abi import describe_local_buffer_abi
from triton.flagmega.codegen.triton.kernel_call_renderers import prepare_kernel_calls


def _binding(formal, descriptor, runtime_argument):
    return {
        "formal": formal,
        "actual": descriptor.id,
        "runtime_argument": runtime_argument,
        "runtime_value_kind": "pointer",
        "abi": describe_local_buffer_abi(descriptor),
    }


def _descriptor(name, dtype, shape, *, offset=0):
    physical = fm.PhysicalBuffer(
        f"physical:{name}", "workspace", fm.dim(65536), 16, fm.dim(offset)
    )
    return fm.BufferDescriptor(
        name,
        dtype,
        shape,
        _dense_strides(shape),
        "workspace",
        16,
        fm.MemSpan(physical, fm.dim(0), fm.dim(_nbytes(dtype, shape))),
    )


def _dense_strides(shape):
    result = []
    stride = 1
    for extent in reversed(shape):
        result.append(stride)
        stride *= extent
    return tuple(reversed(result))


def _nbytes(dtype, shape):
    result = dtype.itemsize
    for extent in shape:
        result *= extent
    return result


def test_local_norm_apply_addresses_outer_rows_mean_scale_bias_and_vector_lanes():
    vector = fm.vector_type("bfloat16", 4)
    source = _descriptor("source", vector, (2, 3, 2))
    stats = _descriptor("stats", fm.DType.FLOAT32, (2, 2, 1, 1))
    scale = _descriptor("scale", vector, (3, 2))
    bias = _descriptor("bias", vector, (3, 2))
    result = _descriptor("result", vector, (2, 3, 2))
    call = prepare_kernel_calls(
        ({
            "call": "apply",
            "family": "norm_apply",
            "variant": "local",
            "execution_kind": "local_shard",
            "parameters": {"block_size": 128},
            "semantic_attrs": {"axis": 1, "epsilon": 1e-5, "use_mean": True},
            "inputs": (
                {"formal": "input", "buffers": (_binding("input", source, "source"),)},
                {"formal": "stats", "buffers": (_binding("stats", stats, "stats"),)},
                {"formal": "scale", "buffers": (_binding("scale", scale, "scale"),)},
                {"formal": "bias", "buffers": (_binding("bias", bias, "bias"),)},
            ),
            "outputs": (
                {"formal": "result", "buffers": (_binding("result", result, "result"),)},
            ),
            "workspaces": (),
        },),
        function_name="main",
    )[0]

    assert call["use_mean"] is True
    assert call["normalization_size"] == 24
    assert call["local_capacity"] == 48
    assert "% 4" in call["lane_coordinate"]
    assert "// 4" in call["source_offset"]
    assert call["stats_sum_offset"] != call["stats_square_sum_offset"]
    assert "norm_apply_local_offsets" in call["scale_offset"]
    assert "norm_apply_local_offsets" in call["bias_offset"]
    assert call["writer_active"] == "True"
