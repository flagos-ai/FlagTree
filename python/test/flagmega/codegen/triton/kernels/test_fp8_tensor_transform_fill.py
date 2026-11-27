# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""FP8 masked loads need floating fill literals, including integer-valued pads."""

from dataclasses import replace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load

from python.test.flagmega.ir.ops.primitive_helpers import primitive_module


@pytest.mark.parametrize("operation,attrs", [
    ("slice", {"starts": (1,), "ends": (257,), "axes": (1,), "steps": (2,)}),
    ("pad", {"pad_end": (0, 3), "pad_value": -2}),
])
@pytest.mark.parametrize("lanes", ((), (2, 4)))
def test_fp8_slice_and_integral_pad_execute_without_integer_fp8_cast(tmp_path, operation, attrs, lanes):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    kind = fm.tensor_type(fm.vector_type("float8_e4m3fn", lanes) if lanes else "float8_e4m3fn", (3, 257))
    graph = replace(primitive_module(fm.get_definition("tensors." + operation), (kind,), **attrs), stage="frozen_constants")
    compiler = Compiler()
    artifact = write_artifact(compiler.compile(graph).module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    values = (torch.arange(3 * 257 * (8 if lanes else 1), device="cuda").float() % 17 - 8).reshape(3, 257, *lanes)
    expected = values[:, 1:257:2] if operation == "slice" else torch.cat((values, values.new_full((3, 3, *lanes), -2)), dim=1)
    values, expected = values.to(torch.float8_e4m3fn), expected.to(torch.float8_e4m3fn)
    runtime = load(artifact, device="cuda:0")
    try:
        runtime.prepare(values)
        actual = runtime.run(values)
        torch.cuda.synchronize()
        assert runtime.resource_report["spill_bytes"] == 0
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    finally:
        runtime.close()
