# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


def _module(*, round_before_scale=False):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (2, 3, 5)))
            scale = self.input("scale", fm.tensor_type("bfloat16", (3, 5)))
            bias = self.input("bias", fm.tensor_type("bfloat16", (3, 5)))
            stats = fm.F.nn.norm_stats(value, axis=1, use_mean=True, name="stats")
            output = fm.F.nn.norm_apply(
                value,
                stats,
                scale,
                bias,
                axis=1,
                epsilon=1e-5,
                use_mean=True,
                round_before_scale=round_before_scale,
                name="output",
            )
            self.function("main", (value, scale, bias), (output,))

    return Graph().build()


def test_generic_norm_apply_source_consumes_mean_bias_and_outer_rows(tmp_path):
    module = Compiler().compile(_module()).module
    artifact = write_artifact(
        module, tmp_path / "generic-norm-apply", target="nvidia-sm90", emit_executable=True
    )
    source = (artifact / "generated_kernels.py").read_text("utf-8")

    assert "# flagmega-kernel: norm_apply/local platform=generic" in source
    assert "norm_apply_mean" in source
    assert "norm_apply_bias" in source
    assert "norm_apply_local_offsets" in source
    assert "tl.maximum(norm_apply_variance, 0.0)" in source


def test_generic_norm_apply_matches_torch_on_h800(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    module = Compiler().compile(_module()).module
    artifact = write_artifact(
        module, tmp_path / "generic-norm-apply-runtime", target="nvidia-sm90", emit_executable=True
    )
    runtime = load(artifact, device="cuda:0")
    value = torch.randn((2, 3, 5), dtype=torch.bfloat16, device="cuda:0")
    scale = torch.randn((3, 5), dtype=torch.bfloat16, device="cuda:0")
    bias = torch.randn((3, 5), dtype=torch.bfloat16, device="cuda:0")
    output = torch.empty_like(value)

    runtime.prepare(value, scale, bias, output=output)
    runtime.run_into(output, value, scale, bias)
    torch.cuda.synchronize()

    expected = torch.nn.functional.layer_norm(
        value.float(), (3, 5), scale.float(), bias.float(), 1e-5
    ).to(torch.bfloat16)
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("round_before_scale", [False, True])
def test_norm_apply_rounding_policy_is_executed_exactly(tmp_path, round_before_scale):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    module = Compiler().compile(_module(round_before_scale=round_before_scale)).module
    artifact = write_artifact(module, tmp_path / "rounding", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    generator = torch.Generator().manual_seed(123)
    value = torch.randn((2, 3, 5), generator=generator).to(device="cuda", dtype=torch.bfloat16)
    scale = torch.randn((3, 5), generator=generator).to(device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros_like(scale)
    source = value.float()
    mean = source.mean(dim=(1, 2), keepdim=True)
    variance = source.square().mean(dim=(1, 2), keepdim=True) - mean.square()
    normalized = (source - mean) * torch.rsqrt(variance + 1e-5)
    rounded = normalized.to(value.dtype).float()
    expected = ((rounded if round_before_scale else normalized) * scale.float()).to(value.dtype)
    # Ensure the test actually distinguishes the policies.
    assert torch.count_nonzero((normalized * scale.float()).to(value.dtype) != (rounded * scale.float()).to(value.dtype))
    output = torch.empty_like(value)
    runtime.prepare(value, scale, bias, output=output)
    runtime.run_into(output, value, scale, bias)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
