# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The bf16 WMMA (mma) dense_matmul renders for non-power-of-two M.

tl.arange and the accumulator shape must be a power of two, but M (prefill token
rows, or a standalone matmul's M=3 / M=17) need not be. The renderer rounds the
M tile up to the next power of two; the existing m_mask (< m_active_extent) drops
the padding rows, so the result is unchanged.
"""

import pytest

from triton.flagmega.codegen.triton.kernel_call_renderers import _dense_matmul_call
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry


def _abi(*, logical_shape, local_shape, scalar_strides, storage_kind="compact_local",
         coordinate_space="local", logical_coordinates=None, component_stride=0):
    if logical_coordinates is None:
        logical_coordinates = tuple(f"local_coord_{axis}" for axis in range(len(logical_shape)))
    return {
        "storage": "workspace",
        "pool_byte_offset": 0,
        "storage_kind": storage_kind,
        "scalar_dtype": "bfloat16",
        "scalar_itemsize": 2,
        "logical_shape": tuple(logical_shape),
        "local_capacity_shape": tuple(local_shape),
        "active_shape_expressions": tuple(str(value) for value in local_shape),
        "logical_coordinate_expressions": tuple(logical_coordinates),
        "scalar_storage_strides": tuple(scalar_strides),
        "scalar_lane_shape": (),
        "scalar_lane_count": 1,
        "component_stride_elements": component_stride,
        "component_stride_scalar_elements": component_stride,
        "coordinate_space": coordinate_space,
    }


def _parameter(formal, abi):
    return {"formal": formal, "buffers": [{"abi": abi, "runtime_argument": "workspace"}]}


def _mma_raw(local_m):
    source_abi = _abi(logical_shape=(local_m, 32), local_shape=(local_m, 32), scalar_strides=(32, 1))
    result_abi = _abi(logical_shape=(local_m, 16), local_shape=(local_m, 16), scalar_strides=(16, 1))
    # transpose_b: weight is (N, K) = (16, 32), canonical global/unpacked.
    weight_abi = _abi(
        logical_shape=(16, 32), local_shape=(16, 32), scalar_strides=(32, 1),
        storage_kind="canonical_global", coordinate_space="canonical_global",
        logical_coordinates=("local_coord_0", "local_coord_1"),
    )
    return {
        "variant": "mma",
        "semantic_op": "ntt.dense_matmul",
        "semantic_attrs": {"transpose_b": True},
        "parameters": {"block_k": 32, "tile_n": 16},
        "inputs": [_parameter("lhs", source_abi), _parameter("rhs", weight_abi)],
        "outputs": [_parameter("result", result_abi)],
    }


@pytest.mark.parametrize("local_m,expected_padded", [(1, 1), (2, 2), (3, 4), (8, 8), (17, 32)])
def test_mma_pads_m_to_the_next_power_of_two(local_m, expected_padded):
    call = _dense_matmul_call(_mma_raw(local_m))
    assert call["local_m_capacity"] == local_m
    assert call["local_m_padded"] == expected_padded


def test_mma_renders_a_power_of_two_arange_for_non_power_of_two_m():
    call = {
        **_dense_matmul_call(_mma_raw(3)),
        "symbol": "_flagmega_test_mma",
        "signature": "lhs, rhs, result",
        "execution_kind": "local",
        "family": "dense_matmul",
        "variant": "mma",
    }
    source = TritonTemplateRegistry().render(
        "kernels/dense_matmul/mma.py.jinja",
        {
            "render_calls": (call,),
            "distributed_entry": True,
            "mesh_axis_names": ("y", "x"),
            "mesh_hierarchy": (8, 16),
        },
    )
    compile(source, "mma_non_pow2.py", "exec")
    # M=3 rounds up to 4 -- Triton rejects tl.arange(0, 3).
    assert "tl.arange(0, 4)" in source
    assert "tl.arange(0, 3)" not in source
    assert "tl.zeros((4, " in source
