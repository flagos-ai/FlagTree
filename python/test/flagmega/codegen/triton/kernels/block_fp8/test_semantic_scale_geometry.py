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


def _parameter(formal, descriptor):
    return {"formal": formal, "buffers": (_binding(formal, descriptor),)}


def test_block_fp8_scale_geometry_comes_from_semantic_weight_format():
    value = _descriptor("value", fm.DType.BFLOAT16, (1, 64))
    weight = _descriptor("weight", fm.DType.FLOAT8_E4M3FN, (64, 64))
    scale = _descriptor("scale", fm.DType.FLOAT32, (2, 2))
    result = _descriptor("result", fm.DType.BFLOAT16, (1, 64))

    call = prepare_kernel_calls(({
        "call": "linear",
        "family": "block_fp8",
        "variant": "simt",
        "execution_kind": "local_shard",
        "parameters": {"tile_n": 16},
        "semantic_attrs": {"weight_block_n": 32, "weight_block_k": 32},
        "inputs": (
            _parameter("value", value),
            _parameter("weight", weight),
            _parameter("weight_scale", scale),
        ),
        "outputs": (_parameter("result", result),),
        "workspaces": (),
    },), function_name="main")[0]

    assert "// 32" in call["scale_row_offset"]
    assert call["block_k"] == 32


def test_block_fp8_catalog_does_not_own_weight_format_geometry():
    from triton.flagmega.targets.nvidia.implementations import (
        sm90_triton_implementation_model,
    )

    model = sm90_triton_implementation_model()
    for name in ("tir.block_fp8.simt", "tir.block_fp8.mma"):
        implementation = model.implementation(name)
        assert implementation is not None
        assert implementation.parameters == {"tile_n": 16}
