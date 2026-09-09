# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Scalar element types do not imply serial single-lane tensor execution."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.targets.nvidia import sm90_triton_implementation_model


@pytest.mark.parametrize("variant", ("add", "mul", "silu", "cast"))
def test_scalar_elementwise_catalog_schedules_a_tensor_tile(variant):
    implementation = next(value for value in sm90_triton_implementation_model().implementations
                          if value.id == f"tir.elementwise.{variant}.scalar")
    assert implementation.contract["vectorization_kind"] == "scalar"
    assert implementation.parameters["elements_per_program"] == 256


def test_unpacked_cast_keeps_scalar_ir_but_emits_parallel_tile(tmp_path):
    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (4096,)), id="value")
            result = fm.F.tensors.cast(value, fm.DType.FLOAT32, name="cast")
            self.function("main", (value,), (result,))

    module = Graph(dialect="nn", stage="frozen_constants", entry="main").build()
    lowered = Compiler().compile(module).module
    package = render_triton_package(lowered, tmp_path / "generated")
    call = next(value for value in package["runtime_binding"]["call_abi"]["kernel_calls"]
                if value["semantic_op"] == "tensors.cast")
    schedule = call["parameters"]["vector_schedule"]
    assert schedule["contract"]["kind"] == "scalar"
    assert schedule["physical"]["elements_per_program"] == 256
    source = (tmp_path / "generated" / "generated_kernels.py").read_text()
    assert "tl.range(0, 4096, 256)" in source
    assert "tl.range(0, 4096, 1)" not in source
