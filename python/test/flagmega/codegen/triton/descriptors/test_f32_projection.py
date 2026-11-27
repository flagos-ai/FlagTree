# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""FP32 projection is the accumulator, not a widened BF16 value."""

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.ir import DType
from triton.flagmega.runtime import load
from .conftest import _packed_norm_stats_pipeline_module


@pytest.mark.parametrize("implementation", [
    "tir.dense_matmul.packed_k_major_gemv_norm_stats",
    "tir.dense_matmul.packed_tensor_descriptor_smem_pipeline_gemv_norm_stats",
    "tir.dense_matmul.packed_tensor_descriptor_table_smem_pipeline_gemv_norm_stats",
])
def test_f32_accumulator_survives_projection_and_residual_cancellation(tmp_path, implementation):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    module = _packed_norm_stats_pipeline_module(implementation, wide=True, projection_output_data_type="float32")
    weights = {
        "weight": torch.zeros((2048, 2048), dtype=torch.bfloat16), "scale": torch.ones(2048, dtype=torch.bfloat16),
        "bias": torch.zeros(2048, dtype=torch.bfloat16)
    }
    weights["weight"][:, 0] = 1
    weights["weight"][:, 1] = 1 / 256
    checkpoint = MemoryCheckpoint(
        {}, {name: TensorInfo(name, DType.BFLOAT16, tuple(value.shape), "memory")
             for name, value in weights.items()}, weights)
    artifact = write_artifact(module, tmp_path / "f32_projection", target="nvidia-sm90", checkpoint=checkpoint,
                              emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    x = torch.zeros((1, 2048), dtype=torch.bfloat16, device="cuda")
    x[:, :2] = 1
    residual = torch.full((1, 2048), -1., dtype=torch.float32, device="cuda")
    # BF16 would erase the half-ULP term and produce zero. F32 must retain it.
    value = x.float() @ weights["weight"].cuda().float().T + residual
    expected = value * (value.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    assert torch.count_nonzero(expected) == expected.numel()
    output = torch.empty_like(residual)
    runtime.prepare(x, residual, output=output)
    runtime.run_into(output, x, residual)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=2e-6, atol=1e-7)
