# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""SIMT and MMA are separate implementations of the same packed QKV ABI."""

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton import describe_tir_package, render_tir_package
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.ir import DType
from triton.flagmega.runtime import load


IMPLEMENTATION = "tir.qkv_parallel_linear.packed_gemv_smem_pipeline"


def test_simt_qkv_keeps_typed_pipe_and_inline_helpers(packed_qkv_simt_pipeline_module):
    package = describe_tir_package(packed_qkv_simt_pipeline_module)
    call = next(call for call in package["render_calls"] if call["family"] == "qkv_parallel_linear")
    assert call["variant"] == "packed_gemv_smem_pipeline"
    assert call["reduction_group"] == 128
    source = render_tir_package(package, "unit")
    assert f"@triton.jit\ndef {call['symbol']}__consumer_stage" in source
    assert "qkv_partial += qkv_weight * qkv_source[None, :]" in source
    assert "tl.dot(" not in source


def test_simt_qkv_matches_three_independent_matmuls(tmp_path, packed_qkv_simt_pipeline_module):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 is required")
    generator = torch.Generator().manual_seed(280)
    weights = {name: torch.randn((2048, n), generator=generator, dtype=torch.bfloat16) / 32
               for name, n in (("q_weight", 2048), ("k_weight", 1024), ("v_weight", 1024))}
    checkpoint = MemoryCheckpoint({}, {
        name: TensorInfo(name, DType.BFLOAT16, tuple(value.shape), "memory") for name, value in weights.items()
    }, weights)
    artifact = write_artifact(packed_qkv_simt_pipeline_module, tmp_path / "qkv", target="nvidia-sm90", checkpoint=checkpoint, emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    value = torch.randn((1, 2048), generator=generator, dtype=torch.bfloat16).cuda() / 8
    outputs = tuple(torch.empty((1, n), dtype=torch.bfloat16, device="cuda") for n in (2048, 1024, 1024))
    runtime.prepare(value, *outputs)
    for _ in range(3):
        for output in outputs:
            output.fill_(float("nan"))
        runtime.run_into(value, *outputs)
        torch.cuda.synchronize()
        for output, weight in zip(outputs, weights.values()):
            torch.testing.assert_close(output, value @ weight.cuda(), atol=.03, rtol=.03)
    assert runtime.resource_report["spill_bytes"] == 0
