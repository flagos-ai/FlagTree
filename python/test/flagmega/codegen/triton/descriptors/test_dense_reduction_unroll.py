# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Dense GEMV must consume the unroll contract its renderer already validates."""

from copy import deepcopy
import pytest

from triton.flagmega.codegen.triton import describe_tir_package, render_tir_package
from triton.flagmega.codegen.triton.kernel_call_renderers import _dense_matmul_call
from triton.flagmega.codegen.triton.runtime_binding import describe_function_runtime_binding
from triton.flagmega.errors import CodegenError


@pytest.mark.parametrize("unroll", (1, 4, 8))
def test_dense_gemv_renders_selected_unroll_and_keeps_helper_inline(packed_descriptor_pipeline_module, unroll):
    package = describe_tir_package(packed_descriptor_pipeline_module)
    calls = [value for value in package["render_calls"] if value["family"] == "dense_matmul"]
    assert calls
    for call in calls:
        call["reduction_unroll"] = unroll
    source = render_tir_package(package, "unit")
    assert f"loop_unroll_factor={unroll}" in source
    assert all(f"@triton.jit\ndef {call['symbol']}__consumer_stage" in source for call in calls)


@pytest.mark.parametrize("unroll", (0, -1, True, 1.5, 3, 64))
def test_dense_gemv_rejects_invalid_unroll(packed_descriptor_pipeline_module, unroll):
    binding = describe_function_runtime_binding(packed_descriptor_pipeline_module)
    raw = deepcopy(next(value for value in binding["call_abi"]["kernel_calls"] if value["family"] == "dense_matmul"))
    raw["parameters"]["reduction_unroll"] = unroll
    with pytest.raises(CodegenError, match="unroll"):
        _dense_matmul_call(raw)
