# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Typed resources and edit/resume validation for bounded Shared LHS caching."""

from copy import deepcopy

import pytest

from .conftest import _packed_norm_stats_pipeline_module
from triton.flagmega.codegen.triton import describe_tir_package, render_tir_package
from triton.flagmega.codegen.triton.kernel_call_renderers import _dense_matmul_norm_stats_call
from triton.flagmega.codegen.triton.runtime_binding import describe_function_runtime_binding
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import emit_module, load_module


IMPLEMENTATION = (
    "tir.dense_matmul."
    "packed_tensor_descriptor_table_smem_pipeline_gemv_norm_stats_lhs8192"
)


@pytest.fixture(scope="module")
def staged_module():
    return _packed_norm_stats_pipeline_module(IMPLEMENTATION, reduction_extent=6144)


def test_staging_uses_typed_consumer_resource_and_round_trips(staged_module, tmp_path):
    path = tmp_path / "staged.py"
    emit_module(staged_module, path)
    resumed = load_module(path)
    package = describe_tir_package(resumed)
    call = next(value for value in package["render_calls"] if value["semantic_op"] == "ntt.matmul_norm_stats")
    assert call["local_k_capacity"] == 6144
    assert call["lhs_stage_extent"] == 8192
    assert call["reduction_group"] == 128
    assert call["reduction_unroll"] == 4
    workspace = call["shared_workspaces"][1]
    assert workspace["name"] == "lhs_stage"
    assert workspace["shape"] == (1, 8192)
    assert workspace["nbytes"] == 16384
    assert workspace["alignment_bytes"] == 1024
    assert tuple(call["pipeline_contract"]["consumer_shared_workspace_indices"]) == (1,)
    source = render_tir_package(package, "unit")
    compile(source, "bounded_lhs.py", "exec")
    assert "for dense_copy_start in tl.range(0, 6144, 1024)" in source
    assert "mask=(dense_local_k_offsets < 6144)" in source
    assert f"@triton.jit\ndef {call['symbol']}__consumer_stage(" in source
    assert "loop_unroll_factor=4" in source


@pytest.mark.parametrize("unroll", [0, -1, 3, 16, True, 1.5])
def test_renderer_rejects_invalid_edited_reduction_unroll(staged_module, unroll):
    binding = describe_function_runtime_binding(staged_module)
    raw = deepcopy(next(value for value in binding["call_abi"]["kernel_calls"] if value["semantic_op"] == "ntt.matmul_norm_stats"))
    raw["parameters"]["reduction_unroll"] = unroll
    with pytest.raises(CodegenError, match="reduction unroll"):
        _dense_matmul_norm_stats_call(raw)


@pytest.mark.parametrize("mutation", ["capacity", "tile", "shape", "dtype", "alignment", "owner"])
def test_renderer_rejects_invalid_edited_staging_resource(staged_module, mutation):
    binding = describe_function_runtime_binding(staged_module)
    raw = deepcopy(next(value for value in binding["call_abi"]["kernel_calls"] if value["semantic_op"] == "ntt.matmul_norm_stats"))
    if mutation == "capacity":
        raw["parameters"]["lhs_stage_extent"] = 4096
    elif mutation == "tile":
        raw["parameters"]["lhs_copy_tile"] = 768
    elif mutation == "shape":
        raw["shared_workspaces"][1]["shape"] = (1, 4096)
    elif mutation == "dtype":
        raw["shared_workspaces"][1]["dtype"] = "float32"
    elif mutation == "alignment":
        raw["shared_workspaces"][1]["alignment_bytes"] = 64
    elif mutation == "owner":
        raw["transfer_pipeline"]["consumer_shared_workspace_indices"] = ()
    with pytest.raises(CodegenError, match="LHS staging"):
        _dense_matmul_norm_stats_call(raw)
