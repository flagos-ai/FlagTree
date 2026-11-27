# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The same local GEMV computes contiguous and affine-strided RHS shards."""

import pytest

from .conftest import _packed_norm_stats_pipeline_module
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton import describe_tir_package
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.runtime import load


@pytest.mark.parametrize("k,n", [(1024, 2048), (4096, 4096), (6144, 2048), (4096, 6144), (8192, 8192)])
def test_affine_owner_table_matches_contiguous_owner_table(tmp_path, k, n):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    generator = torch.Generator().manual_seed(260)
    weights = {
        "weight": torch.randn((n, k), dtype=torch.bfloat16, generator=generator) / 32,
        "scale": torch.ones(n, dtype=torch.bfloat16),
        "bias": torch.zeros(n, dtype=torch.bfloat16),
    }
    checkpoint = MemoryCheckpoint({}, {name: TensorInfo(name, fm.DType.BFLOAT16, tuple(value.shape), "memory") for name, value in weights.items()}, weights)
    value = torch.randn((1, k), dtype=torch.bfloat16, generator=generator).cuda() / 8
    residual = torch.randn((1, n), dtype=torch.bfloat16, generator=generator).cuda() / 8
    results = []
    for name, policy in (("contiguous", fm.SBP.split_contiguous((0, 1), n // 8 // 128)),
                         ("cyclic", fm.SBP.split_block_cyclic((0, 1), 1))):
        module = _packed_norm_stats_pipeline_module(
            "tir.dense_matmul.packed_tensor_descriptor_table_smem_pipeline_gemv_norm_stats",
            reduction_extent=k, output_extent=n, output_policy=policy)
        package = describe_tir_package(module)
        call = next(c for c in package["render_calls"] if c["semantic_op"] == "ntt.matmul_norm_stats")
        assert call["local_k_capacity"] == k
        artifact = write_artifact(module, tmp_path / name, target="nvidia-sm90", checkpoint=checkpoint, emit_executable=True)
        runtime = load(artifact, device="cuda:0")
        output = runtime.create_outputs()
        runtime.prepare(value, residual, output=output)
        runtime.run_into(output, value, residual)
        torch.cuda.synchronize()
        assert runtime.resource_report["spill_bytes"] == 0
        results.append(output.clone())
    # Only the FP32 normalization-statistic reduction tree is repartitioned.
    torch.testing.assert_close(results[1], results[0], rtol=8e-3, atol=2e-3)
    expected_value = value @ weights["weight"].cuda().T + residual
    expected = (expected_value.float() * torch.rsqrt(expected_value.float().square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
    torch.testing.assert_close(results[1], expected, rtol=.04, atol=.04)
