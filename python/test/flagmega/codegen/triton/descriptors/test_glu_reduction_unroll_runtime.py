# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Partial unrolling survives Python resume and both GLU pipe field protocols."""

from dataclasses import replace

import pytest

from .conftest import _packed_glu_descriptor_pipeline_module
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton import describe_tir_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.ir import DType, emit_module, load_module
from triton.flagmega.runtime import load
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.targets.nvidia import sm90_triton_implementation_model


@pytest.mark.parametrize("suffix", (
    "packed_tensor_descriptor_smem_pipeline_inline_gemv",
    "packed_tensor_descriptor_table_paired_smem_pipeline_inline_gemv",
))
@pytest.mark.parametrize("unroll", (1, 4))
def test_glu_partial_unroll_matches_torch_after_resume(tmp_path, suffix, unroll):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    implementation_id = "tir.dense_matmul_glu." + suffix
    model = sm90_triton_implementation_model()
    model = replace(model, implementations=tuple(
        replace(implementation, parameters={**implementation.parameters, "reduction_unroll": unroll})
        if implementation.id == implementation_id else implementation
        for implementation in model.implementations
    ))
    compiler = Compiler()
    compiler.target = NvidiaSm90Target(triton_implementation_model=model)
    module = _packed_glu_descriptor_pipeline_module(implementation_id, compiler=compiler)
    path = tmp_path / "kernel.py"
    emit_module(module, path)
    module = load_module(path)
    call = next(value for value in describe_tir_package(module)["render_calls"]
                if value["family"] == "dense_matmul_glu")
    assert call["reduction_unroll"] == unroll
    generator = torch.Generator().manual_seed(292)
    weights = {name: torch.randn((2048, 2048), generator=generator, dtype=torch.bfloat16) / 32
               for name in ("gate", "up")}
    checkpoint = MemoryCheckpoint({}, {
        name: TensorInfo(name, DType.BFLOAT16, tuple(value.shape), "memory")
        for name, value in weights.items()
    }, weights)
    artifact = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90",
                              checkpoint=checkpoint, emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    value = torch.randn((1, 2048), generator=generator, dtype=torch.bfloat16).cuda() / 8
    expected = torch.nn.functional.silu(value @ weights["gate"].cuda().T) * (value @ weights["up"].cuda().T)
    output = runtime.create_outputs()
    runtime.prepare(value, output=output)
    for _ in range(3):
        output.fill_(float("nan"))
        runtime.run_into(output, value)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=.03, atol=.001)
    assert runtime.resource_report["spill_bytes"] == 0
