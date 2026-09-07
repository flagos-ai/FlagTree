# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Wide/tail pointwise normalization with externally supplied exact statistics."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from triton.flagmega.targets.nvidia import sm90_triton_implementation_model


def test_catalog_does_not_serialize_wide_norm_domains_into_half_cta_tiles():
    model = sm90_triton_implementation_model()
    for name in ("tir.norm_apply.local", "tir.gather_reduce_norm_apply.sum"):
        assert model.implementation(name).parameters["block_size"] == 1024


@pytest.mark.parametrize("extent,use_mean", [(257, False), (2048, True), (4097, False)])
def test_wide_norm_apply_retains_row_stats_tail_and_bf16_rounding(tmp_path, extent, use_mean):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    value_type = fm.tensor_type("bfloat16", (2, extent))
    stats_type = fm.tensor_type("float32", (2 if use_mean else 1, 2, 1))
    parameter_type = fm.tensor_type("bfloat16", (extent,))

    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", value_type)
            stats = self.input("stats", stats_type)
            scale = self.input("scale", parameter_type)
            bias = self.input("bias", parameter_type)
            result = fm.F.nn.norm_apply(value, stats, scale, bias, axis=1,
                epsilon=1e-5, use_mean=use_mean, round_before_scale=True)
            self.function("main", (value, stats, scale, bias), (result,))

    module = Compiler().compile(Graph(dialect="high_level", stage="imported", entry="main").build()).module
    artifact = write_artifact(module, tmp_path / "wide", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    generator = torch.Generator().manual_seed(272)
    value = torch.randn((2, extent), generator=generator).bfloat16().cuda()
    scale = torch.randn(extent, generator=generator).bfloat16().cuda()
    bias = torch.zeros_like(scale)
    square_sum = value.float().square().sum(-1, keepdim=True)
    total = value.float().sum(-1, keepdim=True)
    stats = torch.stack((total, square_sum)) if use_mean else square_sum.unsqueeze(0)
    mean = total / extent if use_mean else 0.0
    variance = square_sum / extent - mean * mean
    normalized = ((value.float() - mean) * torch.rsqrt(variance.clamp_min(0) + 1e-5)).bfloat16()
    expected = normalized * scale
    output = torch.empty_like(value)
    runtime.prepare(value, stats, scale, bias, output=output)
    runtime.run_into(output, value, stats, scale, bias)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert runtime.resource_report["spill_bytes"] == 0
