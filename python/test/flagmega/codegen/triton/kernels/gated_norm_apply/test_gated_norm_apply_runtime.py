# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The generated kernel must keep the gate in FP32 and round exactly once.

The expanded ``norm_apply`` + ``sigmoid`` + ``mul`` + ``cast`` chain rounds twice
(the normalized value to the input dtype, the activation to the gate dtype).
These tests pin the fused kernel against ``FusedRMSNormGated.forward_native`` at
``atol=0``, so re-introducing either boundary in the template fails.
"""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load

TOKENS, HEADS, DIM = 2, 3, 5
EPS = 1e-5


def _module(*, activation="sigmoid"):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (TOKENS, HEADS, DIM)))
            gate = self.input("gate", fm.tensor_type("bfloat16", (TOKENS, HEADS, DIM)))
            scale = self.input("scale", fm.tensor_type("bfloat16", (DIM,)))
            bias = self.input("bias", fm.tensor_type("bfloat16", (DIM,)))
            stats = fm.F.nn.norm_stats(value, axis=-1, use_mean=False, name="stats")
            output = fm.F.nn.gated_norm_apply(
                value, stats, scale, bias, gate,
                axis=-1, epsilon=EPS, use_mean=False, activation=activation,
                name="output",
            )
            self.function("main", (value, gate, scale, bias), (output,))

    return Graph().build()


def _reference(value, gate, scale, bias, activation):
    """vLLM ``FusedRMSNormGated.forward_native``, no-residual branch."""
    import torch

    wide = value.float()
    variance = wide.pow(2).mean(dim=-1, keepdim=True)
    normalized = wide * torch.rsqrt(variance + EPS) * scale.float() + bias.float()
    gate_wide = gate.float()
    if activation == "sigmoid":
        activated = torch.sigmoid(gate_wide)
    else:
        activated = gate_wide * torch.sigmoid(gate_wide)
    return (normalized * activated).to(value.dtype)


def test_generated_source_keeps_one_rounding_boundary(tmp_path):
    module = Compiler().compile(_module()).module
    artifact = write_artifact(
        module, tmp_path / "gated-norm-apply", target="nvidia-sm90", emit_executable=True
    )
    source = (artifact / "generated_kernels.py").read_text("utf-8")

    assert "# flagmega-kernel: gated_norm_apply/local platform=generic" in source
    assert "gated_norm_apply_gate" in source
    assert "tl.sigmoid(gated_norm_apply_gate)" in source
    # The activation and the normalized value stay fp32; only the store converts.
    assert "gated_norm_apply_activated.to(" not in source
    assert "gated_norm_apply_output.to(tl.bfloat16)" in source


@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
def test_generated_kernel_is_bit_exact_against_the_reference(tmp_path, activation):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    module = Compiler().compile(_module(activation=activation)).module
    artifact = write_artifact(
        module, tmp_path / f"gated-{activation}", target="nvidia-sm90", emit_executable=True
    )
    runtime = load(artifact, device="cuda:0")
    generator = torch.Generator().manual_seed(7)
    value = torch.randn((TOKENS, HEADS, DIM), generator=generator).to("cuda", torch.bfloat16)
    gate = torch.randn((TOKENS, HEADS, DIM), generator=generator).to("cuda", torch.bfloat16)
    scale = torch.randn((DIM,), generator=generator).to("cuda", torch.bfloat16)
    bias = torch.zeros_like(scale)
    expected = _reference(value, gate, scale, bias, activation)
    output = torch.empty_like(value)

    runtime.prepare(value, gate, scale, bias, output=output)
    runtime.run_into(output, value, gate, scale, bias)
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_rounding_the_gate_activation_would_be_observable(tmp_path):
    """Guard against a vacuous pass: the two policies must actually differ."""
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(7)
    value = torch.randn((TOKENS, HEADS, DIM), generator=generator).to(torch.bfloat16)
    gate = torch.randn((TOKENS, HEADS, DIM), generator=generator).to(torch.bfloat16)
    scale = torch.randn((DIM,), generator=generator).to(torch.bfloat16)
    bias = torch.zeros_like(scale)
    wide = value.float()
    normalized = (
        wide * torch.rsqrt(wide.pow(2).mean(dim=-1, keepdim=True) + EPS) * scale.float()
        + bias.float()
    )
    fused = (normalized * torch.sigmoid(gate.float())).to(torch.bfloat16)
    rounded_gate = torch.sigmoid(gate.float()).to(torch.bfloat16).float()
    rounded_norm = normalized.to(torch.bfloat16).float()
    expanded = (rounded_norm * rounded_gate).to(torch.bfloat16)
    assert torch.count_nonzero(fused != expanded), (
        "This fixture no longer separates the rounding policies, so the bit-exact "
        "runtime assertion above would pass on a doubly-rounded kernel."
    )
