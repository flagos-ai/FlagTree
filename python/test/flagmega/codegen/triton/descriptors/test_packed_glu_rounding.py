# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Gate projection, up projection and activation each publish a BF16 value."""

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.ir import DType
from triton.flagmega.runtime import load


@pytest.mark.parametrize("fixture_name", [
    "packed_glu_portable_pipeline_module",
    "packed_glu_direct_lhs_descriptor_pipeline_module",
    "packed_glu_descriptor_pipeline_module",
    "packed_glu_paired_table_inline_descriptor_pipeline_module",
])
def test_packed_glu_keeps_all_three_rounding_boundaries(tmp_path, request, fixture_name):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    module = request.getfixturevalue(fixture_name)
    gate = torch.zeros((2048, 2048), dtype=torch.bfloat16)
    up = torch.zeros_like(gate)
    gate[:, 0] = 1
    up[:, 0] = 1
    # Output classes isolate gate rounding, up rounding and SiLU rounding.
    gate[0::3, 1] = 1 / 256
    up[1::3, 1] = 1 / 256
    up[2::3, 0] = 129 / 128
    weights = {"gate": gate, "up": up}
    checkpoint = MemoryCheckpoint({}, {
        name: TensorInfo(name, DType.BFLOAT16, tuple(value.shape), "memory")
        for name, value in weights.items()
    }, weights)
    artifact = write_artifact(module, tmp_path / "rounding", target="nvidia-sm90",
                              checkpoint=checkpoint, emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    value = torch.zeros((1, 2048), dtype=torch.bfloat16, device="cuda")
    value[:, :2] = 1
    g = gate[:, :2].float().sum(-1).cuda()
    u = up[:, :2].float().sum(-1).cuda()
    expected = torch.nn.functional.silu(g.bfloat16()) * u.bfloat16()
    unrounded = (torch.nn.functional.silu(g) * u).bfloat16()
    for offset in range(3):
        assert torch.count_nonzero(expected[offset::3] != unrounded[offset::3])
    output = torch.empty_like(value)
    runtime.prepare(value, output=output)
    runtime.run_into(output, value)
    torch.cuda.synchronize()
    assert runtime.resource_report["spill_bytes"] == 0
    torch.testing.assert_close(output.flatten(), expected, rtol=0, atol=0)
