# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Asynchronous copies must preserve bytes across repeated consumer calls."""

import re

import pytest

from .conftest import _packed_norm_stats_pipeline_module
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.ir import DType
from triton.flagmega.runtime import load


BASE = "tir.dense_matmul.packed_tensor_descriptor_table_smem_pipeline_gemv_norm_stats_lhs8192"


@pytest.mark.parametrize("k", (1024, 2048, 6144, 8192))
def test_async_lhs_is_bitwise_equal_to_sync_across_changed_inputs(tmp_path, k):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    generator = torch.Generator().manual_seed(329)
    weight = torch.randn((2048, k), dtype=torch.bfloat16, generator=generator) / 32
    # Exercise the BF16 projection boundary independently of the paired run.
    weight[:16] = 0
    weight[:16, 0] = 1
    weight[:16, 1] = 1 / 256
    weights = {"weight": weight, "scale": torch.ones((2048,), dtype=torch.bfloat16),
               "bias": torch.zeros((2048,), dtype=torch.bfloat16)}
    checkpoint = MemoryCheckpoint({}, {
        name: TensorInfo(name, DType.BFLOAT16, tuple(value.shape), "memory")
        for name, value in weights.items()
    }, weights)
    runtimes = []
    for suffix in ("", "_async"):
        module = _packed_norm_stats_pipeline_module(BASE + suffix, reduction_extent=k)
        artifact = write_artifact(module, tmp_path / (suffix or "sync"), target="nvidia-sm90",
                                  checkpoint=checkpoint, emit_executable=True)
        runtimes.append(load(artifact, device="cuda:0"))
    outputs = [runtime.create_outputs() for runtime in runtimes]
    value = torch.empty((1, k), dtype=torch.bfloat16, device="cuda")
    residual = torch.empty((1, 2048), dtype=torch.bfloat16, device="cuda")
    for runtime, output in zip(runtimes, outputs, strict=True):
        runtime.prepare(value, residual, output=output)
    for _ in range(3):
        value.copy_(torch.randn((1, k), dtype=torch.bfloat16, generator=generator) / 8)
        residual.copy_(torch.randn((1, 2048), dtype=torch.bfloat16, generator=generator) / 8)
        value[:, :2] = 1
        residual[:, :16] = -1
        for runtime, output in zip(runtimes, outputs, strict=True):
            output.fill_(float("nan"))
            runtime.run_into(output, value, residual)
        torch.cuda.synchronize()
        torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
        assert torch.count_nonzero(outputs[1][:, :16]) == 0
        expected_value = value @ weight.cuda().T + residual
        expected = (expected_value.float() * torch.rsqrt(expected_value.float().square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
        torch.testing.assert_close(outputs[1], expected, rtol=.04, atol=.04)
    assert all(runtime.resource_report["spill_bytes"] == 0 for runtime in runtimes)
    ptx = runtimes[1]._prepared.compiled_kernel.asm["ptx"]
    assert re.search(r"cp\.async\.(?:ca|cg)\.shared\.global", ptx)
