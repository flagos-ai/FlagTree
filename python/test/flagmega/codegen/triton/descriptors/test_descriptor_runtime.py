# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
# Runtime integration for compiler-produced tensor descriptors.

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.runtime import load

@pytest.mark.parametrize("transpose_b", [False, True])
def test_reused_descriptor_gemv_matches_torch_on_sm90(
    tmp_path, transpose_b, two_call_descriptor_module,
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")

    artifact = write_artifact(
        two_call_descriptor_module(transpose_b=transpose_b),
        tmp_path / "descriptor-gemv",
        target="nvidia-sm90",
        emit_executable=True,
    )
    runtime = load(artifact, device="cuda:0")
    value = torch.randn((1, 128), dtype=torch.bfloat16, device="cuda:0")
    first_weight = torch.randn(
        (128, 128), dtype=torch.bfloat16, device="cuda:0") / 8
    second_weight = torch.randn(
        (128, 128), dtype=torch.bfloat16, device="cuda:0") / 8
    output = torch.empty((1, 128), dtype=torch.bfloat16, device="cuda:0")

    runtime.prepare(value, first_weight, second_weight, output=output)
    runtime.run_into(output, value, first_weight, second_weight)
    torch.cuda.synchronize()

    first_rhs = first_weight.T if transpose_b else first_weight
    second_rhs = second_weight.T if transpose_b else second_weight
    expected = value @ first_rhs @ second_rhs
    assert runtime.prepare_count == 1
    assert runtime.resource_report["spill_bytes"] == 0
    torch.testing.assert_close(output, expected, rtol=3e-2, atol=3e-2)
