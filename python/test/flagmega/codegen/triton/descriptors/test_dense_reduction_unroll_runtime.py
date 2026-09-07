# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Exact arithmetic checks loop unrolling across stage and output-tile tails."""

from dataclasses import replace
import pytest

from .conftest import _packed_descriptor_pipeline_module
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton import describe_tir_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.ir import DType, emit_module, load_module
from triton.flagmega.runtime import load
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.targets.nvidia import sm90_triton_implementation_model


@pytest.mark.parametrize("suffix,k,n", (
    ("packed_tensor_descriptor_smem_pipeline_gemv", 1024, 8192),
    ("packed_tensor_descriptor_table_smem_pipeline_gemv_tn64_bk512", 2048, 9216),
))
@pytest.mark.parametrize("unroll", (1, 4))
def test_partial_unroll_exact_dyadic_projection_after_resume(tmp_path, suffix, k, n, unroll):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    implementation_id = "tir.dense_matmul." + suffix
    model = sm90_triton_implementation_model()
    model = replace(model, implementations=tuple(replace(value, parameters={**value.parameters, "reduction_unroll": unroll})
        if value.id == implementation_id else value for value in model.implementations))
    compiler = Compiler()
    compiler.target = NvidiaSm90Target(triton_implementation_model=model)
    module = _packed_descriptor_pipeline_module(implementation_id, k=k, n=n, compiler=compiler)
    path = tmp_path / "kernel.py"
    emit_module(module, path)
    module = load_module(path)
    call = next(value for value in describe_tir_package(module)["render_calls"] if value["family"] == "dense_matmul")
    assert call["reduction_unroll"] == unroll
    weight = ((torch.arange(n)[:, None] * 5 + torch.arange(k)[None, :] * 3) % 61 - 30).to(torch.bfloat16)
    checkpoint = MemoryCheckpoint({}, {"weight": TensorInfo("weight", DType.BFLOAT16, (n, k), "memory")}, {"weight": weight})
    value = ((torch.arange(k).double() % 17 - 8) / 128).reshape(1, k)
    expected = (value @ weight.double().T).to(torch.bfloat16).cuda()
    value = value.to(device="cuda", dtype=torch.bfloat16)
    runtime = load(write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", checkpoint=checkpoint, emit_executable=True), device="cuda:0")
    output = runtime.create_outputs()
    runtime.prepare(value, output=output)
    for _ in range(3):
        output.fill_(float("nan"))
        runtime.run_into(output, value)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert runtime.resource_report["spill_bytes"] == 0
