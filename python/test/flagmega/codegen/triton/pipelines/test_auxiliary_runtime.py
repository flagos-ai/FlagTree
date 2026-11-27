# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.ir import DType
from triton.flagmega.runtime import load

_AUX_PIPELINE = (
    "tir.dense_matmul.tensor_descriptor_smem_pipeline_aux_gemv"
)


def _checkpoint(weights):
    return MemoryCheckpoint(
        {},
        {
            name: TensorInfo(name, DType.BFLOAT16, (128, 128), "memory")
            for name in weights
        },
        weights,
    )


@pytest.mark.parametrize("reusable", (False, True), ids=("direct", "reusable"))
def test_auxiliary_pipeline_matches_torch_on_sm90(
    tmp_path, compile_pipeline_module, reusable,
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")

    first_weight = torch.randn((128, 128), dtype=torch.bfloat16) / 8
    weights = {"weight": first_weight}
    if reusable:
        weights = {
            "first_weight": first_weight,
            "second_weight": torch.randn(
                (128, 128), dtype=torch.bfloat16
            ) / 8,
        }
    artifact = write_artifact(
        compile_pipeline_module(
            reusable=reusable,
            implementation=_AUX_PIPELINE,
        ),
        tmp_path / "dense-auxiliary-pipeline",
        target="nvidia-sm90",
        checkpoint=_checkpoint(weights),
        emit_executable=True,
    )
    runtime = load(artifact, device="cuda:0")
    value = torch.randn((1, 128), dtype=torch.bfloat16, device="cuda:0")
    output = torch.empty_like(value)
    expected = value @ first_weight.to("cuda:0").T
    if reusable:
        expected = expected @ weights["second_weight"].to("cuda:0").T

    runtime.prepare(value, output=output)
    runtime.run_into(output, value)
    torch.cuda.synchronize()

    assert runtime.prepare_count == 1
    assert runtime.resource_report["spill_bytes"] == 0
    assert runtime.resource_report["shared_memory_bytes"] >= 8192
    torch.testing.assert_close(output, expected, rtol=3e-2, atol=3e-2)
