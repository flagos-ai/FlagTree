# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit FP32 carry and sequential FMA must survive actual device lowering."""

from math import prod

import pytest
import torch
from triton.flagmega.runtime import load

from python.test.flagmega.codegen.triton.kernels.mhc.test_mhc_runtime import _artifact, _run, _skip_without_sm90
from python.test.flagmega.ir.ops.nn.test_mhc_residual_precision import CollapseCarry, PostCarry


@pytest.mark.parametrize("hc,hidden", [(4, 8), (3, 17)])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_post_preserves_value_first_fma_and_store_dtype(tmp_path, hc, hidden, dtype):
    _skip_without_sm90(torch)
    host = {"value": torch.full((1, hidden), 1.5, dtype=torch.bfloat16),
            "residual": torch.zeros((1, hc, hidden), dtype=torch.bfloat16),
            "post": torch.full((1, hc), 1 + 2 ** -23), "comb": torch.zeros((1, hc, hc))}
    host["residual"][:, 0] = -1.5
    host["comb"][:, 0] = 1.
    artifact = _artifact(tmp_path, "carry", PostCarry(hc, hidden, dtype).build())
    source = (artifact / "generated_kernels.py").read_text()
    assert "tl.fma(" in source
    runtime = load(artifact, device="cuda:0")
    try:
        values = {name: value.cuda() for name, value in host.items()}
        output = torch.empty((1, hc, hidden), dtype=getattr(torch, dtype), device="cuda")
        _run(runtime, values, {"carry": output}, torch)
        torch.testing.assert_close(output.cpu(), torch.full_like(output.cpu(), 3 * 2 ** -24), rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


def test_post_keeps_ascending_source_order_on_device(tmp_path):
    _skip_without_sm90(torch)
    host = {"value": torch.zeros((1, 8), dtype=torch.bfloat16),
            "residual": torch.ones((1, 4, 8), dtype=torch.bfloat16),
            "post": torch.zeros((1, 4)),
            "comb": torch.tensor([2 ** 24, 1., -(2 ** 24), 2.]).view(1, 4, 1).expand(1, 4, 4).contiguous()}
    runtime = load(_artifact(tmp_path, "ordered-carry", PostCarry().build()), device="cuda:0")
    try:
        values = {name: value.cuda() for name, value in host.items()}
        output = torch.empty((1, 4, 8), device="cuda")
        _run(runtime, values, {"carry": output}, torch)
        torch.testing.assert_close(output.cpu(), torch.full((1, 4, 8), 2.), rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("lanes", [(), (8,), (2, 4)])
def test_pre_rounds_only_the_collapse_input_and_keeps_sequential_fma(tmp_path, lanes):
    _skip_without_sm90(torch)
    streams = torch.zeros((1, 4, 8))
    streams[:, 0] = 1.5 + 2 ** -20
    streams[:, 1] = -1.5
    host = {"streams": streams, "projection": torch.zeros((24, 32)),
            "base": torch.zeros(24), "scale": torch.zeros(3)}
    artifact = _artifact(tmp_path, "collapse", CollapseCarry(lanes=lanes).build())
    source = (artifact / "generated_kernels.py").read_text()
    assert "# flagmega-kernel: pack/" not in source
    runtime = load(artifact, device="cuda:0")
    try:
        values = {name: value.cuda() for name, value in host.items()}
        outputs = {"field_0": torch.empty((1, 4), device="cuda"),
                   "field_1": torch.empty((1, 4, 4), device="cuda"),
                   "field_2": torch.empty((1, 8 // prod(lanes), *lanes), device="cuda")}
        _run(runtime, values, outputs, torch)
        torch.testing.assert_close(outputs["field_2"].cpu(), torch.full_like(outputs["field_2"].cpu(), 2 ** -25),
                                   rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
