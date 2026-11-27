# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""SM90 execution coverage for packed projection/residual/RMS-statistics."""

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.ir import DType
from triton.flagmega.runtime import load


def test_packed_norm_stats_pipeline_matches_torch_on_sm90(
    tmp_path,
    packed_norm_stats_descriptor_pipeline_module,
):
    torch = pytest.importorskip("torch")
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (9, 0)
    ):
        pytest.skip("SM90 CUDA is required")

    generator = torch.Generator().manual_seed(20260904)
    weight = torch.randn(
        (2048, 2048), generator=generator, dtype=torch.bfloat16
    ) / 32
    scale = torch.randn(
        (2048,), generator=generator, dtype=torch.bfloat16
    ) / 8 + 1
    bias = torch.randn(
        (2048,), generator=generator, dtype=torch.bfloat16
    ) / 64
    checkpoint = MemoryCheckpoint(
        {},
        {
            name: TensorInfo(
                name,
                DType.BFLOAT16,
                tuple(value.shape),
                "memory",
            )
            for name, value in {
                "weight": weight,
                "scale": scale,
                "bias": bias,
            }.items()
        },
        {"weight": weight, "scale": scale, "bias": bias},
    )
    artifact = write_artifact(
        packed_norm_stats_descriptor_pipeline_module,
        tmp_path / "packed-dense-norm-stats",
        target="nvidia-sm90",
        checkpoint=checkpoint,
        emit_executable=True,
    )
    runtime = load(artifact, device="cuda:0")
    value = (
        torch.randn((1, 2048), generator=generator, dtype=torch.bfloat16)
        .to("cuda:0")
        / 8
    )
    residual = (
        torch.randn((1, 2048), generator=generator, dtype=torch.bfloat16)
        .to("cuda:0")
        / 8
    )
    residual_before = residual.clone()
    output = runtime.create_outputs()

    expected_value = (value @ weight.to("cuda:0").T + residual_before).to(
        torch.bfloat16
    )
    square_sum = expected_value.float().square().sum(dim=-1, keepdim=True)
    expected = (
        expected_value.float()
        * torch.rsqrt(square_sum / 2048 + 1e-6)
        * scale.to("cuda:0").float()
        + bias.to("cuda:0").float()
    ).to(torch.bfloat16)

    runtime.prepare(value, residual, output=output)
    runtime.run_into(output, value, residual)
    torch.cuda.synchronize()

    assert runtime.prepare_count == 1
    assert runtime.resource_report["spill_bytes"] == 0
    assert runtime.resource_report["shared_memory_bytes"] >= 131072
    torch.testing.assert_close(output, expected, rtol=4e-2, atol=4e-2)
