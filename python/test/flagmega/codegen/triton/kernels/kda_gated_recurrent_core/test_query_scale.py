# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Scaled KDA readout must survive packing without scaling recurrent state."""

import pytest
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

from python.test.flagmega.codegen.triton.kernels.kda_gated_recurrent_core.test_kda_gated_recurrent_runtime import (
    _module,
)
from python.test.flagmega.devices import is_nvidia_sm90


@pytest.mark.parametrize("gated", (False, True))
@pytest.mark.parametrize("l2norm", (False, True))
@pytest.mark.parametrize("packing", ("scalar", "direct", "demand"))
def test_scaled_readout_keeps_the_same_state_after_optional_normalization(tmp_path, gated, l2norm, packing):
    torch = pytest.importorskip("torch")
    if not is_nvidia_sm90():
        pytest.skip("SM90 CUDA is required")
    vector = torch.tensor([[[1., 0., 0., 0.]]]).repeat(1, 2, 1).bfloat16()
    value = torch.tensor([[[2., 4., 6., 8.]]]).repeat(1, 2, 1).bfloat16()
    state = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1) / 4
    host = {"q": vector, "k": vector.clone(), "v": value, "state": state}
    if gated:
        # The saturated forget sigmoid makes exp(decay) round to one. Beta's
        # sigmoid is exactly 1/2, leaving a nonzero state to test independently.
        host.update(gate=torch.full_like(vector, -100), decay_rate=torch.ones(1, 2, 4),
                    gate_bias=torch.zeros(1, 2, 4), beta=torch.zeros(1, 2, dtype=torch.bfloat16))
    else:
        host.update(decay=torch.zeros(1, 2, 4), beta=torch.full((1, 2), .5))
    packing_attrs = ({"output_lanes": (2,)} if packing == "direct" else
                     {"consumer_lanes": (2,)} if packing == "demand" else {})
    module = _module(gated=gated, l2norm=l2norm, query_scale=.5, **packing_attrs)
    path = write_artifact(Compiler().compile(module).module, tmp_path / "scaled",
                          target="nvidia-sm90", emit_executable=True)
    assert "# flagmega-kernel: pack/" not in (path / "generated_kernels.py").read_text()
    runtime = load(path, device="cuda:0")
    try:
        values = {name: tensor.cuda() for name, tensor in host.items()}
        expected = torch.tensor([[[.5625, 1., 1.5, 2.]]]).repeat(1, 2, 1).bfloat16()
        shape = (1, 2, 4) if packing == "scalar" else (1, 2, 2, 2)
        buffers = (torch.empty(shape, dtype=torch.bfloat16, device="cuda"),
                   torch.empty_like(state, device="cuda"))
        specs = [spec for spec in runtime.external_arguments if spec["role"] == "result"]
        outputs = {spec["value"]: buffer for spec, buffer in zip(specs, buffers, strict=True)}
        arguments = tuple(outputs[spec["value"]] if spec["role"] == "result" else values[spec["value"]]
                          for spec in runtime.external_arguments)
        GeneratedTirCallGraphModule.prepare(runtime, *arguments)
        GeneratedTirCallGraphModule.run_into(runtime, *arguments)
        torch.testing.assert_close(buffers[0].cpu().reshape(expected.shape), expected, rtol=0, atol=0)
        key = vector.float()
        if l2norm:
            key = key / torch.sqrt(key.square().sum(-1, keepdim=True) + 1e-6)
        correction = (state * key.unsqueeze(-1)).sum(-2)
        delta = (value.float() - correction) * .5
        expected_state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
        torch.testing.assert_close(buffers[1].cpu(), expected_state, rtol=1e-6, atol=1e-7)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
