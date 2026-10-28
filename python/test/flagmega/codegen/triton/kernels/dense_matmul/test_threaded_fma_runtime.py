# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A packed FP32 projection preserves its declared per-lane FMA order."""
from dataclasses import replace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.importer import DirectoryCheckpoint
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule
from triton.flagmega.targets import NvidiaSm90Target

from python.test.flagmega.codegen.triton.kernels.dense_matmul.test_packed_mma_runtime import CanonicalTarget
from python.test.flagmega.devices import is_nvidia_sm90


def ordered_reference(value, weight):
    lanes = torch.arange(128)
    result = []
    for x in value:
        accumulator = torch.zeros((weight.shape[0], 128), dtype=torch.float32)
        for tile in range(value.shape[1] // 1024):
            for v in range(8):
                k = tile * 1024 + lanes * 8 + v
                accumulator = (accumulator.double() + weight[:, k].double() * x[k].double()).float()
        for step in (16, 8, 4, 2, 1):
            accumulator = accumulator + accumulator[:, lanes ^ step]
        result.append(((accumulator[:, 0] + accumulator[:, 32]) + accumulator[:, 64]) + accumulator[:, 96])
    return torch.stack(result)


@pytest.mark.parametrize("m,k,n", [(1, 4096, 24), (2, 1024, 8)])
@pytest.mark.parametrize("warps", [4, 8])
def test_packed_threaded_fma_preserves_fp32_dot_order(tmp_path, m, k, n, warps):
    if not is_nvidia_sm90():
        pytest.skip("SM90 CUDA is required")
    from safetensors.torch import save_file

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (m, k)), id="x")
            w = self.weight("weight", fm.tensor_type("bfloat16", (n, k)),
                            source="model.safetensors", key="weight", id="weight")
            y = fm.F.math.matmul(x, w, transpose_b=True, output_data_type="float32")
            self.function("main", (x,), (y,))

    torch.manual_seed(129)
    weight, value = torch.randn(n, k).bfloat16(), torch.randn(m, k).bfloat16()
    expected = ordered_reference(value, weight)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n")
    save_file({"weight": weight}, checkpoint / "model.safetensors")
    base = NvidiaSm90Target().triton_implementation_model
    identity = "tir.dense_matmul.packed_k_major_threaded_fma"
    model = replace(base, implementations=tuple(
        replace(i, parameters={**i.parameters, "compute_num_warps": warps}) if i.id == identity else i
        for i in base.implementations), preferences={**base.preferences, "dense_matmul":
                    (identity, *base.preferences["dense_matmul"])})
    compiler = Compiler()
    compiler.target = CanonicalTarget(triton_implementation_model=model)
    result = compiler.compile(Graph(dialect="high_level", stage="imported", entry="main",
                                    metadata={"floating_point_contract": "strict"}).build()).module
    artifact = write_artifact(result, tmp_path / "artifact", target="nvidia-sm90",
                              checkpoint=DirectoryCheckpoint(checkpoint), emit_executable=True)
    source = (artifact / "generated_kernels.py").read_text()
    assert "# flagmega-kernel: dense_matmul/packed_k_major_threaded_fma" in source
    assert "# flagmega-kernel: pack/" not in source
    runtime = load(artifact, device="cuda:0")
    try:
        actual = torch.empty((m, n), dtype=torch.float32, device="cuda")
        x = value.cuda()
        args = tuple(actual if s["role"] == "result" else x for s in runtime.external_arguments)
        GeneratedTirCallGraphModule.prepare(runtime, *args)
        GeneratedTirCallGraphModule.run_into(runtime, *args)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
