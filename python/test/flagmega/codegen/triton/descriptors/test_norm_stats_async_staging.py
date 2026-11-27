# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit asynchronous LHS staging: source legality and completion order."""

import ast
from copy import deepcopy

import pytest

from .conftest import _packed_norm_stats_pipeline_module
from triton.flagmega.codegen.triton import describe_tir_package, render_tir_package
from triton.flagmega.codegen.triton.kernel_call_renderers import _dense_matmul_norm_stats_call
from triton.flagmega.codegen.triton.runtime_binding import describe_function_runtime_binding
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import emit_module, load_module


BASE = "tir.dense_matmul.packed_tensor_descriptor_table_smem_pipeline_gemv_norm_stats_lhs8192"
ASYNC = BASE + "_async"


@pytest.fixture(scope="module")
def sync_raw():
    module = _packed_norm_stats_pipeline_module(BASE, reduction_extent=6144)
    return next(value for value in describe_function_runtime_binding(module)["call_abi"]["kernel_calls"]
                if value["semantic_op"] == "ntt.matmul_norm_stats")


@pytest.mark.parametrize("k", (1024, 2048, 6144, 8192))
def test_async_staging_is_explicit_and_completes_before_first_read(tmp_path, k):
    module = _packed_norm_stats_pipeline_module(ASYNC, reduction_extent=k)
    path = tmp_path / "async.py"
    emit_module(module, path)
    package = describe_tir_package(load_module(path))
    call = next(value for value in package["render_calls"] if value["semantic_op"] == "ntt.matmul_norm_stats")
    assert call["lhs_copy_kind"] == "async"
    source = render_tir_package(package, "unit")
    function = next(value for value in ast.parse(source).body
                    if isinstance(value, ast.FunctionDef) and value.name == call["symbol"] + "__consumer")
    body = ast.get_source_segment(source, function)
    assert "is_async=True" in body
    assert "tl.store(dense_copy_pointer" not in body
    assert f"tl.static_range(0, {k}, 1024)" in body
    assert body.index("async_commit_group") < body.index(".wait(dense_pipeline_sequence)")
    assert body.index(".wait(dense_pipeline_sequence)") < body.index("async_wait_group")
    assert body.index("async_wait_group") < body.index("tl.debug_barrier") < body.index("__consumer_stage(")


@pytest.mark.parametrize("mutation", ("mode", "bool_mode", "tail", "stride", "pool_offset", "scope_stride", "owner_stride", "external"))
def test_async_staging_rejects_an_unproven_physical_row(sync_raw, mutation):
    raw = deepcopy(sync_raw)
    raw["parameters"]["lhs_copy_kind"] = "async"
    abi = raw["inputs"][0]["buffers"][0]["abi"]
    if mutation == "mode":
        raw["parameters"]["lhs_copy_kind"] = "automatic"
    elif mutation == "bool_mode":
        raw["parameters"]["lhs_copy_kind"] = True
    elif mutation == "tail":
        raw["parameters"]["lhs_copy_tile"] = 4096
    elif mutation == "stride":
        abi["scalar_storage_strides"] = (12288, 2)
    elif mutation == "pool_offset":
        abi["pool_byte_offset"] = 2
    elif mutation == "scope_stride":
        abi["pool_scope_stride_bytes"] += 2
    elif mutation == "owner_stride":
        abi["storage_kind"] = "compact_per_owner"
        abi["component_stride_scalar_elements"] = 6145
    elif mutation == "external":
        abi["pooled"] = False
    with pytest.raises(CodegenError, match="LHS (copy|async)"):
        _dense_matmul_norm_stats_call(raw)


def test_sync_candidate_keeps_its_masked_copy_contract(sync_raw):
    raw = deepcopy(sync_raw)
    raw["parameters"]["lhs_copy_tile"] = 4096
    call = _dense_matmul_norm_stats_call(raw)
    assert call["lhs_copy_kind"] == "sync"
    assert call["local_k_capacity"] == 6144
