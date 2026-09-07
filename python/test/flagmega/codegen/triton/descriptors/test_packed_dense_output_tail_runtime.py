# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Exercise production TMA / pipe output tails with independent dense oracles."""

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.ir import DType, SBP
from triton.flagmega.runtime import load

from .conftest import _packed_descriptor_pipeline_module


@pytest.mark.parametrize("n,cyclic", [(17408, False), (17408, True), (17416, True), (32768, False)])
def test_output_tail_pipeline_matches_torch(tmp_path, n, cyclic):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 is required")
    implementation = "tir.dense_matmul.packed_tensor_descriptor_table_smem_pipeline_gemv_tn64_bk512"
    policy = SBP.split_block_cyclic((0, 1), 1) if cyclic else SBP.split_contiguous((0, 1), n // 8 // 128)
    module = _packed_descriptor_pipeline_module(implementation, n=n, output_policy=policy)
    generator = torch.Generator().manual_seed(276)
    weight = torch.randn((n, 2048), dtype=torch.bfloat16, generator=generator) / 32
    checkpoint = MemoryCheckpoint({}, {"weight": TensorInfo("weight", DType.BFLOAT16, tuple(weight.shape), "memory")}, {"weight": weight})
    artifact = write_artifact(module, tmp_path / "tail", target="nvidia-sm90", checkpoint=checkpoint, emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    value = torch.randn((1, 2048), dtype=torch.bfloat16, generator=generator).cuda() / 8
    output = torch.empty((1, n), dtype=torch.bfloat16, device="cuda")
    runtime.prepare(value, output=output)
    expected = value @ weight.cuda().T
    for _ in range(3):
        output.fill_(float("nan"))
        runtime.run_into(output, value)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, atol=.03, rtol=.03)
    assert runtime.resource_report["spill_bytes"] == 0
