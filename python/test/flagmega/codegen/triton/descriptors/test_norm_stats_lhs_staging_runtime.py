# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Shared-LHS and direct-LHS variants must preserve the same arithmetic."""

import pytest

from .conftest import _packed_norm_stats_pipeline_module
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.ir import DType
from triton.flagmega.runtime import load


BASE = "tir.dense_matmul.packed_tensor_descriptor_table_smem_pipeline_gemv_norm_stats"


@pytest.mark.parametrize("reduction_extent", [1024, 4096, 6144, 8192])
def test_shared_lhs_matches_direct_lhs_exactly_with_capacity_padding(tmp_path, reduction_extent):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    generator = torch.Generator().manual_seed(257)
    weight = torch.randn((2048, reduction_extent), dtype=torch.bfloat16, generator=generator) / 32
    # Also exercise the previously repaired BF16 projection/residual boundary.
    weight[:16] = 0
    weight[:16, 0] = 1
    weight[:16, 1] = 1 / 256
    weights = {
        "weight": weight,
        "scale": torch.ones((2048,), dtype=torch.bfloat16),
        "bias": torch.zeros((2048,), dtype=torch.bfloat16),
    }
    checkpoint = MemoryCheckpoint({}, {
        name: TensorInfo(name, DType.BFLOAT16, tuple(value.shape), "memory")
        for name, value in weights.items()
    }, weights)
    value = torch.randn((1, reduction_extent), dtype=torch.bfloat16, generator=generator).cuda() / 8
    value[:, :2] = 1
    residual = torch.randn((1, 2048), dtype=torch.bfloat16, generator=generator).cuda() / 8
    residual[:, :16] = -1
    outputs = []
    for suffix in ("", "_lhs8192"):
        module = _packed_norm_stats_pipeline_module(BASE + suffix, reduction_extent=reduction_extent)
        artifact = write_artifact(module, tmp_path / (suffix or "direct"), target="nvidia-sm90", checkpoint=checkpoint, emit_executable=True)
        runtime = load(artifact, device="cuda:0")
        output = runtime.create_outputs()
        runtime.prepare(value, residual, output=output)
        runtime.run_into(output, value, residual)
        torch.cuda.synchronize()
        assert runtime.resource_report["spill_bytes"] == 0
        if suffix:
            assert runtime.resource_report["shared_memory_bytes"] >= 147456
        outputs.append(output.clone())
    torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
    assert torch.count_nonzero(outputs[1][:, :16]) == 0
    expected_value = value @ weight.cuda().T + residual
    expected = (expected_value.float() * torch.rsqrt(expected_value.float().square().mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
    torch.testing.assert_close(outputs[1], expected, rtol=4e-2, atol=4e-2)
