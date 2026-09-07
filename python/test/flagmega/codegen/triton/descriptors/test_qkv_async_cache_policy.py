# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit, user-authorized workaround for ptxas cp.async cache-hint wrong code.

Keep this separate from numerical QKV tests: the failure also occurs without
matrix arithmetic. See build/flagmega/ptxas_cache_policy_bug.md for raw PTX.
"""

import ast

import pytest

from triton.flagmega.codegen.triton import describe_tir_package, render_tir_package


@pytest.mark.parametrize("fixture", (
    "packed_qkv_mma_pipeline_module",
    "packed_qkv_mma_descriptor_table_pipeline_module",
    "packed_qkv_simt_pipeline_module",
))
def test_input_copy_keeps_async_without_unsafe_cache_hint(request, fixture):
    package = describe_tir_package(request.getfixturevalue(fixture))
    call = next(value for value in package["render_calls"] if value["family"] == "qkv_parallel_linear")
    tree = ast.parse(render_tir_package(package, "unit"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    consumer = functions[call["symbol"] + "__consumer"]
    producer = functions[call["symbol"] + "__producer"]
    assert ast.unparse(consumer.decorator_list[0]) == "triton.jit(noinline=True)"
    assert ast.unparse(producer.decorator_list[0]) == "triton.jit(noinline=True)"
    copies = [node for node in ast.walk(consumer)
              if isinstance(node, ast.Call) and ast.unparse(node.func) == "tle.gpu.copy"]
    assert len(copies) == 1
    keywords = {item.arg: ast.literal_eval(item.value) for item in copies[0].keywords}
    assert keywords["is_async"] is True
    assert "eviction_policy" not in keywords
    assert "tle.gpu.async_commit_group()" in ast.unparse(consumer)
    assert "tle.gpu.async_wait_group(0)" in ast.unparse(consumer)
    # TMA is a different instruction family; its streaming hint is unaffected.
    assert 'eviction_policy=\'evict_first\'' in ast.unparse(producer)
