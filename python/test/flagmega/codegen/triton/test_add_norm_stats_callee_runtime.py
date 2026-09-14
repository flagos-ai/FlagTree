# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Owner-local residual statistics returned through a reusable device function."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from ...passes.auto_distributed.test_function_return_publication import residual_callee


def test_distributed_residual_callee_returns_value_and_materialized_stats(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    module = Compiler().compile(residual_callee()).module
    artifact = write_artifact(module, tmp_path / "residual-callee", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    first = torch.linspace(-2, 2, 2048, device="cuda", dtype=torch.bfloat16).reshape(1, 256, 8)
    second = torch.full_like(first, .125)
    originals = first.clone(), second.clone()
    expected = (torch.nn.functional.silu(first) + second).bfloat16()
    expected_stats = expected.float().square().sum()
    values = {"first": first, "second": second}
    for argument in runtime.external_arguments:
        buffer = runtime.buffer_plan.buffer_map[argument["buffer"]]
        if buffer.id not in values:
            lanes = buffer.dtype.lanes if isinstance(buffer.dtype, fm.VectorType) else ()
            dtype = buffer.dtype.elem_type if lanes else buffer.dtype
            values[buffer.id] = torch.empty((*buffer.shape, *lanes), dtype=getattr(torch, dtype.value), device="cuda")
    arguments = tuple(values[argument["buffer"]] for argument in runtime.external_arguments)
    output_bindings = runtime.buffer_plan.function_map[runtime.ir_module.entry].outputs
    outputs = tuple(values[buffer] for _, buffers in output_bindings for buffer in buffers)
    assert len(outputs) == 2
    runtime.prepare(*arguments)
    for _ in range(3):
        for value in outputs:
            value.fill_(float("nan"))
        first.copy_(originals[0])
        second.copy_(originals[1])
        runtime.run_into(*arguments)
        torch.cuda.synchronize()
        torch.testing.assert_close(outputs[0].reshape(-1), expected.reshape(-1), rtol=.01, atol=.01)
        torch.testing.assert_close(outputs[1].reshape(()), expected_stats, rtol=.01, atol=.01)
