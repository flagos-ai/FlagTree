# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

from triton.flagmega.codegen.triton.kernel_call_renderers import _dense_matmul_call
from triton.flagmega.codegen.triton.templates import (
    KernelTemplateSpec,
    TritonTemplateRegistry,
)


def _abi(
    logical_shape,
    local_shape,
    scalar_strides,
    *,
    scalar_lane_shape=(),
):
    lane_count = 1
    for extent in scalar_lane_shape:
        lane_count *= extent
    return {
        "storage": "rdata",
        "pool_byte_offset": 512,
        "storage_kind": "canonical_global",
        "scalar_dtype": "bfloat16",
        "scalar_itemsize": 2,
        "logical_shape": tuple(logical_shape),
        "local_capacity_shape": tuple(local_shape),
        "active_shape_expressions": tuple(str(value) for value in local_shape),
        "logical_coordinate_expressions": tuple(
            f"local_coord_{axis}" for axis in range(len(logical_shape))
        ),
        "scalar_storage_strides": tuple(scalar_strides),
        "scalar_lane_shape": tuple(scalar_lane_shape),
        "scalar_lane_count": lane_count,
        "component_stride_elements": 0,
        "component_stride_scalar_elements": 0,
        "coordinate_space": "canonical_global",
    }


def _parameter(name, abi):
    return {
        "formal": name,
        "buffers": [{"abi": abi, "runtime_argument": "rdata"}],
    }


def _raw_call():
    source = _abi((1, 2048), (1, 2048), (2048, 1))
    weight = _abi(
        (128, 256),
        (128, 256),
        (32768, 128),
        scalar_lane_shape=(8, 2, 8),
    )
    result = _abi(
        (1, 256),
        (1, 256),
        (2048, 8),
        scalar_lane_shape=(8,),
    )
    return {
        "semantic_op": "ntt.packed_matmul",
        "variant": "packed_tensor_descriptor_smem_pipeline_gemv",
        "semantic_attrs": {"transpose_b": True},
        "parameters": {
            "packed_layout": "k_major_n8_k16",
            "block_k": 1024,
            "tile_n": 16,
            "num_stages": 3,
            "reduction_group": 32,
            "consumer_warps": 8,
            "worker_width": 32,
            "producer_warps": 1,
            "producer_registers": 24,
        },
        "inputs": [_parameter("lhs", source), _parameter("rhs", weight)],
        "outputs": [_parameter("result", result)],
        "shared_workspaces": ({"name": "weight_stage"},),
        "transfer_pipeline": {"channels": ({"name": "weight"},)},
    }


def test_packed_descriptor_reinterprets_vector_storage_as_physical_tma_axes():
    call = _dense_matmul_call(_raw_call())
    request = call["host_tensor_descriptor_requests"][0]

    assert request == {
        "parameter": "weight_descriptor",
        "source": "rdata",
        "kind": "single",
        "offset_bytes": 512,
        "dtype": "bfloat16",
        "shape": (128, 256, 2, 64),
        "strides": (32768, 128, 64, 1),
        "block_shape": (64, 2, 2, 64),
        "source_shape_axes": ((), (), (), ()),
        "padding": "zero",
    }
    assert len(call["descriptor_offsets"]) == 4
    assert "dense_local_k_start" in call["descriptor_offsets"][0]
    assert call["descriptor_offsets"][0].endswith("// 16), tl.int32)")
    assert "dense_local_n_start" in call["descriptor_offsets"][1]
    assert call["descriptor_offsets"][1].endswith("// 8), tl.int32)")
    assert call["descriptor_offsets"][2:] == (
        "tl.full((), 0, tl.int32)",
        "tl.full((), 0, tl.int32)",
    )
    assert call["descriptor_block_shape"] == (64, 2, 2, 64)


def test_packed_descriptor_pipeline_template_maps_physical_tile_to_local_nk():
    registry = TritonTemplateRegistry()
    spec = KernelTemplateSpec(
        "dense_matmul",
        "packed_tensor_descriptor_smem_pipeline_gemv",
        "nvidia",
        "sm90",
    )
    source = registry.render_kernel(spec, {}).source
    template = Path(registry.root, registry.resolve(spec)).read_text()

    compile(source, "packed_tensor_descriptor_smem_pipeline_gemv.py", "exec")
    assert "tle.gpu.copy(" in template
    assert "call.descriptor_offsets" in template
    assert "dense_stage_k // 16" in template
    assert "dense_local_n // 8" in template
    assert "tle.gpu.BlockEncoding(" in template
    assert "loop_unroll_factor={{ call.reduction_unroll }}" in template
    assert "qwen" not in source.lower()
