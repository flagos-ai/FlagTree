# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Coarse head projections execute scalar and packed result contracts."""

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

from python.test.flagmega.ir.ops.nn.test_mla_projection import graph
from python.test.flagmega.devices import requires_nvidia_sm90


@requires_nvidia_sm90
@pytest.mark.parametrize("direction", ["absorb_query", "expand_value"])
@pytest.mark.parametrize("lanes", [(), (2, 4)])
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("placement", [None, fm.Placement((8, 16), "yx", "bb")])
def test_device_projection_preserves_interleaved_rows_and_lanes(tmp_path, direction, lanes, fp8, placement):
    source = graph(direction, lanes, input_lanes=(2, 4), fp8=fp8, tokens=1, placement=placement)
    compiled = Compiler(CompileOptions(placement=placement)).compile(source).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    width = 16 if direction == "absorb_query" else 32
    x = ((torch.arange(2 * width, device="cuda") % 13) - 6).float().mul_(0.125).bfloat16().reshape(1, 2, width)
    w = ((torch.arange(48 * 32, device="cuda") % 17) - 8).float().mul_(0.125).bfloat16().reshape(48, 32)
    values = {"x": x.reshape(1, 2, width // 8, 2, 4), "w": w}
    if fp8:
        values.update(w=w.to(torch.float8_e4m3fn), scale=torch.full((6, 2), 0.5, device="cuda"))
        w = (values["w"].float() * 0.5).bfloat16()
    selected = w.reshape(2, 24, 32)
    selected = selected[:, :16] if direction == "absorb_query" else selected[:, 16:].transpose(1, 2)
    expected = torch.bmm(x.transpose(0, 1).float(), selected.float()).bfloat16().transpose(0, 1)
    output = torch.empty_like(expected)
    sentinel_output = torch.empty(128, dtype=torch.float32, device="cuda") if placement is not None else None
    if placement is not None:
        values["sentinel"] = torch.arange(128, dtype=torch.float32, device="cuda")
    result_values = iter((output, sentinel_output))
    values.update(
        (spec["value"], next(result_values)) for spec in runtime.external_arguments if spec["role"] == "result"
    )
    try:
        arguments = tuple(values[spec["value"]] for spec in runtime.external_arguments)
        GeneratedTirCallGraphModule.prepare(runtime, *arguments)
        GeneratedTirCallGraphModule.run_into(runtime, *arguments)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
        if placement is not None:
            assert runtime.resource_report["grid"] == [128, 1, 1]
            torch.testing.assert_close(sentinel_output, values["sentinel"] * 2, rtol=0, atol=0)
    finally:
        runtime.close()
