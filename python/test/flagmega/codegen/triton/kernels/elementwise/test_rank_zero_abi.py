# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Rank-zero by-value parameters are not the scalar-element tensor ABI.

The existing scalar entry arithmetic limitation must stay an explicit error;
changing the tensor tile must not reinterpret scalar values as pointers.
"""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import CodegenError
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.targets.nvidia import sm90_triton_implementation_model


@pytest.mark.parametrize("tile", (1, 256))
@pytest.mark.parametrize("variant", ("add", "mul", "silu", "cast"))
def test_rank_zero_arithmetic_is_not_silently_lowered_as_tensor_pointer(tmp_path, tile, variant):
    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", ()), id="value")
            if variant == "cast":
                output = fm.F.tensors.cast(value, fm.DType.FLOAT32, name="output")
            elif variant == "silu":
                output = fm.F.math.silu(value, name="output")
            else:
                output = getattr(fm.F.math, variant)(value, value, name="output")
            self.function("main", (value,), (output,))

    model = sm90_triton_implementation_model()
    implementations = tuple(
        replace(value, parameters={"elements_per_program": tile})
        if value.id == f"tir.elementwise.{variant}.scalar" else value for value in model.implementations)
    compiler = Compiler()
    compiler.target = NvidiaSm90Target(triton_implementation_model=replace(model, implementations=implementations))
    final = compiler.compile(Graph(dialect="nn", stage="frozen_constants", entry="main").build()).module
    with pytest.raises(CodegenError, match="scalar values are not buffer pointers"):
        render_triton_package(final, tmp_path / "generated")
