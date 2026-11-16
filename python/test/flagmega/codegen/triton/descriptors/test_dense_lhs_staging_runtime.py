# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Real registered async transport preserves exact BF16 across row reuse."""

import re

import pytest

from .test_dense_lhs_staging import BASE, REGISTERED, make_module
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.ir import DType
from triton.flagmega.runtime import load


def test_registered_plain_async_row_matches_original_exactly_on_reuse(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    k, n = 6144, 2048
    weight = (((torch.arange(n)[:, None] * 5 + torch.arange(k)[None, :] * 3) % 15 - 7) / 16).bfloat16()
    weight[:16] = 0
    checkpoint = MemoryCheckpoint({}, {
        "weight": TensorInfo("weight", DType.BFLOAT16, (n, k), "memory"),
    }, {"weight": weight})
    runtimes = [load(write_artifact(make_module(implementation), tmp_path / name,
                                    target="nvidia-sm90", checkpoint=checkpoint,
                                    emit_executable=True), device="cuda:0")
                for name, implementation in (("ordinary", BASE), ("async", REGISTERED))]
    value = torch.empty((1, k), dtype=torch.bfloat16, device="cuda")
    outputs = [runtime.create_outputs() for runtime in runtimes]
    for runtime, output in zip(runtimes, outputs, strict=True):
        runtime.prepare(value, output=output)
    for iteration in range(4):
        # Products and all six-stage FP32 partial sums are exactly dyadic.
        row = (((torch.arange(k) * (iteration + 1) + iteration * 7) % 17 - 8) / 64).reshape(1, k)
        if iteration == 2:
            row.zero_()
        value.copy_(row)
        expected = ((row.double() * 2) @ weight.double().T).bfloat16().cuda()
        for runtime, output in zip(runtimes, outputs, strict=True):
            output.fill_(float("nan"))
            runtime.run_into(output, value)
        torch.cuda.synchronize()
        torch.testing.assert_close(outputs[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
        assert not torch.isnan(outputs[1]).any()
    ptx = runtimes[1]._prepared.compiled_kernel.asm["ptx"]
    assert re.search(r"cp\.async\.(?:ca|cg)\.shared\.global", ptx)
