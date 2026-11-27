# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Selected GLU unrolling is a checked schedule parameter, not ignored metadata."""

from copy import deepcopy

import pytest

from triton.flagmega.codegen.triton.kernel_call_renderers import _dense_matmul_glu_call
from triton.flagmega.codegen.triton.runtime_binding import describe_function_runtime_binding
from triton.flagmega.codegen.triton.tir_package import describe_tir_package, render_tir_package
from triton.flagmega.errors import CodegenError


def _raw(module):
    binding = describe_function_runtime_binding(module)
    return deepcopy(next(value for value in binding["call_abi"]["kernel_calls"]
                         if value["semantic_op"] == "nn.packed_dense_matmul_glu"))


@pytest.mark.parametrize("unroll", (1, 4, 16, 32))
def test_glu_accepts_exact_divisor_unroll(packed_glu_paired_table_inline_descriptor_pipeline_module, unroll):
    raw = _raw(packed_glu_paired_table_inline_descriptor_pipeline_module)
    raw["parameters"]["reduction_unroll"] = unroll
    assert _dense_matmul_glu_call(raw)["reduction_unroll"] == unroll


@pytest.mark.parametrize("unroll", (0, -1, True, 1.5, 3, 64))
def test_glu_rejects_invalid_unroll(packed_glu_paired_table_inline_descriptor_pipeline_module, unroll):
    raw = _raw(packed_glu_paired_table_inline_descriptor_pipeline_module)
    raw["parameters"]["reduction_unroll"] = unroll
    with pytest.raises(CodegenError, match="unroll"):
        _dense_matmul_glu_call(raw)


def test_glu_default_retains_full_unroll(packed_glu_paired_table_inline_descriptor_pipeline_module):
    raw = _raw(packed_glu_paired_table_inline_descriptor_pipeline_module)
    call = _dense_matmul_glu_call(raw)
    assert call["reduction_unroll"] == call["reduction_groups_per_stage"]


def test_glu_template_consumes_unroll_without_helper_call_boundary(packed_glu_paired_table_inline_descriptor_pipeline_module):
    package = describe_tir_package(packed_glu_paired_table_inline_descriptor_pipeline_module)
    calls = tuple(value for value in package["render_calls"] if value["family"] == "dense_matmul_glu")
    assert calls
    for call in calls:
        call["reduction_unroll"] = 4
    source = render_tir_package(package, "unit")
    compile(source, "glu-unroll.py", "exec")
    assert "loop_unroll_factor=4" in source
    for call in calls:
        assert f"@triton.jit\ndef {call['symbol']}__consumer_stage" in source
