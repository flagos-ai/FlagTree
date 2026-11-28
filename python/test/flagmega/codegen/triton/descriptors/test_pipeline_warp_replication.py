# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A shared entry can increase compute warps without invalidating GEMV layouts."""

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.ir import DType
from triton.flagmega.runtime import load
from triton.flagmega.targets import NvidiaSm90Target
from .conftest import _packed_descriptor_pipeline_module


def test_packed_pipeline_in_sixteen_warp_entry(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    target = NvidiaSm90Target()
    ordinary_launch = target.launch_planner
    target.launch_planner = lambda module, nodes: {**ordinary_launch(module, nodes), "num_warps": 16}
    compiler = Compiler()
    compiler.target = target
    module = _packed_descriptor_pipeline_module(compiler=compiler)
    k = n = 2048
    weight = ((torch.arange(n)[:, None] + torch.arange(k)[None, :]) % 7 - 3).bfloat16()
    checkpoint = MemoryCheckpoint({}, {"weight": TensorInfo("weight", DType.BFLOAT16, (n, k), "memory")},
                                  {"weight": weight})
    artifact = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", checkpoint=checkpoint,
                              emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    try:
        value = ((torch.arange(k) % 11 - 5) / 128).reshape(1, k).bfloat16().cuda()
        expected = (value.cpu().double() @ weight.double().T).bfloat16().cuda()
        output = runtime.create_outputs()
        runtime.prepare(value, output=output)
        assert runtime.resource_report["num_warps"] >= 16
        for _ in range(2):
            output.fill_(float("nan"))
            runtime.run_into(output, value)
            torch.cuda.synchronize()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
    finally:
        runtime.close()
