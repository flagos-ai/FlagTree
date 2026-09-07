# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""SM90 numerical coverage for the packed BF16 QKV MMA pipeline."""

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.ir import DType
from triton.flagmega.runtime import load


def test_packed_qkv_mma_pipeline_matches_torch_on_sm90(
    tmp_path,
    packed_qkv_mma_pipeline_module,
):
    torch = pytest.importorskip("torch")
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (9, 0)
    ):
        pytest.skip("SM90 CUDA is required")

    generator = torch.Generator().manual_seed(20260904)
    weights = {
        "q_weight": torch.randn(
            (2048, 2048), generator=generator, dtype=torch.bfloat16
        ) / 32,
        "k_weight": torch.randn(
            (2048, 1024), generator=generator, dtype=torch.bfloat16
        ) / 32,
        "v_weight": torch.randn(
            (2048, 1024), generator=generator, dtype=torch.bfloat16
        ) / 32,
    }
    checkpoint = MemoryCheckpoint(
        {},
        {
            name: TensorInfo(
                name,
                DType.BFLOAT16,
                tuple(value.shape),
                "memory",
            )
            for name, value in weights.items()
        },
        weights,
    )
    artifact = write_artifact(
        packed_qkv_mma_pipeline_module,
        tmp_path / "packed-qkv-mma",
        target="nvidia-sm90",
        checkpoint=checkpoint,
        emit_executable=True,
    )
    runtime = load(artifact, device="cuda:0")
    value = (
        torch.randn((1, 2048), generator=generator, dtype=torch.bfloat16)
        .to("cuda:0")
        / 8
    )
    outputs = (
        torch.empty((1, 2048), dtype=torch.bfloat16, device="cuda:0"),
        torch.empty((1, 1024), dtype=torch.bfloat16, device="cuda:0"),
        torch.empty((1, 1024), dtype=torch.bfloat16, device="cuda:0"),
    )
    expected = tuple(
        value @ weights[name].to("cuda:0")
        for name in ("q_weight", "k_weight", "v_weight")
    )

    runtime.prepare(value, *outputs)
    runtime.run_into(value, *outputs)
    torch.cuda.synchronize()

    assert runtime.prepare_count == 1
    assert runtime.resource_report["spill_bytes"] == 0
    assert runtime.resource_report["shared_memory_bytes"] >= 66048
    for actual, reference in zip(outputs, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=4e-2, atol=4e-2)
