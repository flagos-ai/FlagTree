# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""SM90 execution coverage for packed dense TMA staging."""

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.ir import DType
from triton.flagmega.runtime import load


def test_packed_descriptor_pipeline_matches_torch_on_sm90(
    tmp_path,
    packed_descriptor_pipeline_module,
):
    torch = pytest.importorskip("torch")
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (9, 0)
    ):
        pytest.skip("SM90 CUDA is required")

    generator = torch.Generator().manual_seed(20260904)
    weight = torch.randn(
        (2048, 2048), generator=generator, dtype=torch.bfloat16
    ) / 32
    checkpoint = MemoryCheckpoint(
        {},
        {
            "weight": TensorInfo(
                "weight",
                DType.BFLOAT16,
                (2048, 2048),
                "memory",
            )
        },
        {"weight": weight},
    )
    artifact = write_artifact(
        packed_descriptor_pipeline_module,
        tmp_path / "packed-dense-pipeline",
        target="nvidia-sm90",
        checkpoint=checkpoint,
        emit_executable=True,
    )
    runtime = load(artifact, device="cuda:0")
    value = torch.randn(
        (1, 2048), generator=generator, dtype=torch.bfloat16
    ).to("cuda:0") / 8
    output = torch.empty((1, 2048), dtype=torch.bfloat16, device="cuda:0")

    runtime.prepare(value, output=output)
    runtime.run_into(output, value)
    torch.cuda.synchronize()

    expected = value @ weight.to("cuda:0").T
    assert runtime.prepare_count == 1
    assert runtime.resource_report["spill_bytes"] == 0
    assert runtime.resource_report["shared_memory_bytes"] >= 131072
    torch.testing.assert_close(output, expected, rtol=3e-2, atol=3e-2)
