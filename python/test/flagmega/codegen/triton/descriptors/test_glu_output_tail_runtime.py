# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""TMA tail tiles preserve the typed footprint and close every issued epoch."""

import pytest
from .conftest import _packed_glu_descriptor_pipeline_module
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton import describe_tir_package, render_tir_package
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.ir import DType, emit_module, load_module
from triton.flagmega.runtime import load


@pytest.mark.parametrize("n", (1024, 3072))
@pytest.mark.parametrize("suffix", (
    "packed_tensor_descriptor_smem_pipeline_inline_gemv",
    "packed_tensor_descriptor_paired_smem_pipeline_inline_gemv",
    "packed_tensor_descriptor_table_paired_smem_pipeline_inline_gemv",
))
def test_glu_tail_roundtrip_exact_output_and_full_pipe_close(tmp_path, n, suffix):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    module = _packed_glu_descriptor_pipeline_module("tir.dense_matmul_glu." + suffix, output_extent=n)
    path = tmp_path / "tail.py"
    emit_module(module, path)
    module = load_module(path)
    package = describe_tir_package(module)
    call = next(value for value in package["render_calls"] if value["family"] == "dense_matmul_glu")
    assert call["local_n_capacity"] == n // 128
    assert call["tile_n"] == 16
    assert all(value["shape"][-3:] == (2, 2, 64) for value in call["shared_workspaces"])
    source = render_tir_package(package, "unit")
    assert f"tl.cdiv({n // 128}, 16)" in source
    weights = {name: torch.zeros((n, 2048), dtype=torch.bfloat16) for name in ("gate", "up")}
    rows = torch.arange(n)
    weights["gate"][:, 0] = (rows % 13 - 6) / 8
    weights["gate"][:, 1] = 1 / 256
    weights["gate"][:, -1] = 1 / 128
    weights["up"][:, 0] = (rows % 7 - 3) / 8
    weights["up"][:, 1] = 1 / 128
    weights["up"][:, -1] = 1 / 512
    checkpoint = MemoryCheckpoint({}, {name: TensorInfo(name, DType.BFLOAT16, tuple(value.shape), "memory")
                                       for name, value in weights.items()}, weights)
    value = torch.zeros((1, 2048), dtype=torch.bfloat16, device="cuda")
    value[:, [0, 1, 2047]] = 1
    projections = [(value.cpu().double() @ weights[name].double().T).to(device="cuda", dtype=torch.bfloat16)
                   for name in ("gate", "up")]
    expected = torch.nn.functional.silu(projections[0]) * projections[1]
    runtime = load(write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", checkpoint=checkpoint,
                                  emit_executable=True), device="cuda:0")
    output = runtime.create_outputs()
    runtime.prepare(value, output=output)
    for _ in range(3):
        output.fill_(float("nan"))
        runtime.run_into(output, value)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert runtime.resource_report["spill_bytes"] == 0
