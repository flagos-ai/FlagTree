# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A wide epilogue retains BF16 projection ties in every kernel variant."""

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
@pytest.mark.parametrize("round_addend", [False, True])
def test_wide_residual_preserves_projection_rounding_and_fp32_sum(tmp_path, implementation, round_addend):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    module = _packed_norm_stats_pipeline_module(implementation, wide=True,
        addend_cast_dtypes=("bfloat16", "float32") if round_addend else ())
    weights = {"weight": torch.zeros((2048, 2048), dtype=torch.bfloat16),
               "scale": torch.ones(2048, dtype=torch.bfloat16), "bias": torch.zeros(2048, dtype=torch.bfloat16)}
    weights["weight"][:, 0] = 1
    weights["weight"][:, 1] = 1 / 256
    checkpoint = MemoryCheckpoint({}, {name: TensorInfo(name, DType.BFLOAT16, tuple(value.shape), "memory")
                                       for name, value in weights.items()}, weights)
    artifact = write_artifact(module, tmp_path / "wide_residual", target="nvidia-sm90", checkpoint=checkpoint,
                              emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    x = torch.zeros((1, 2048), dtype=torch.bfloat16, device="cuda")
    x[:, :2] = 1
    residual = torch.full((1, 2048), -1., dtype=torch.float32, device="cuda")
    residual[:, 1::2] += .001
    # FP32 1+1/256 is a BF16 tie that rounds to 1 BEFORE adding the residual.
    expected_value = 1. + (residual.bfloat16().float() if round_addend else residual)
    expected = expected_value * (expected_value.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    output = torch.empty_like(residual)
    runtime.prepare(x, residual, output=output)
    runtime.run_into(output, x, residual)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=2e-6, atol=1e-7)
