# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Partial Sum materialization: all-mesh and both hybrid reduction axes."""

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.runtime import load
from .conftest import partial_sum_module


@pytest.mark.parametrize("values,axes,use_mean", [
    (1, (0, 1), False), (3, (0, 1), True),
    (25, (0,), False), (17, (1,), True),
])
def test_partial_reduce_preserves_group_owners_and_ragged_output(tmp_path, values, axes, use_mean):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    module, owners = partial_sum_module(values, axes, use_mean=use_mean)
    artifact = write_artifact(module, tmp_path / "partial", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    source = (torch.arange(values * owners, device="cuda").reshape(values, owners) % 13 - 6).float()
    sum_sq = source.square().sum(-1, keepdim=True)
    expected = torch.stack((source.sum(-1, keepdim=True), sum_sq)) if use_mean else sum_sq.unsqueeze(0)
    output = torch.full_like(expected, float("nan"))
    runtime.prepare(source, output=output)
    # Repeated launches also exercise scratch/collective synchronization reuse.
    for _ in range(3):
        runtime.run_into(output, source)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert runtime.resource_report["spill_bytes"] == 0
