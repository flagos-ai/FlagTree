# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A BF16 matmul result is rounded before the separate residual addition."""

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.ir import DType
from triton.flagmega.runtime import load


@pytest.mark.parametrize("fixture_name", [
    "packed_norm_stats_portable_pipeline_module",
    "packed_norm_stats_descriptor_pipeline_module",
    "packed_norm_stats_descriptor_table_pipeline_module",
])
def test_fused_packed_projection_keeps_bf16_rounding_before_cancellation(tmp_path, request, fixture_name):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    module = request.getfixturevalue(fixture_name)
    weight = torch.zeros((2048, 2048), dtype=torch.bfloat16)
    weight[:, 0] = 1
    weight[:, 1] = 1 / 256
    weights = {
        "weight": weight,
        "scale": torch.ones((2048,), dtype=torch.bfloat16),
        "bias": torch.zeros((2048,), dtype=torch.bfloat16),
    }
    checkpoint = MemoryCheckpoint({}, {
        name: TensorInfo(name, DType.BFLOAT16, tuple(value.shape), "memory")
        for name, value in weights.items()
    }, weights)
    artifact = write_artifact(
        module, tmp_path / "rounding", target="nvidia-sm90", checkpoint=checkpoint,
        emit_executable=True,
    )
    runtime = load(artifact, device="cuda:0")
    value = torch.zeros((1, 2048), dtype=torch.bfloat16, device="cuda")
    value[:, :2] = 1
    residual = torch.full_like(value, -1)
    # FP32 dot is 1 + 2^-8, exactly halfway between BF16 1 and 1 + 2^-7.
    # The materialized matmul rounds to even (1), then residual cancels it.
    expected_value = value @ weight.cuda().T + residual
    assert torch.count_nonzero(expected_value) == 0
    output = runtime.create_outputs()
    runtime.prepare(value, residual, output=output)
    runtime.run_into(output, value, residual)
    torch.cuda.synchronize()
    assert runtime.resource_report["spill_bytes"] == 0
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
