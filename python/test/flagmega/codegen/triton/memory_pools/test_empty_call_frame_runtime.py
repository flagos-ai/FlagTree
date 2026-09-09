# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Exercise the zero-byte pool ABI on device, including a block mesh."""

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime import load
from .test_empty_call_frame import empty_chain


@pytest.mark.parametrize("level", ["fast", "optimized"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("distributed", [False, True])
def test_empty_call_frames_execute_without_access(tmp_path, level, nested, distributed):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    module = Compiler(CompileOptions(bufferize_opt_level=level)).compile(
        empty_chain(nested=nested, distributed=distributed)).module
    runtime = load(write_artifact(module, tmp_path / "empty", target="nvidia-sm90", emit_executable=True), device="cuda:0")
    tensors = {}
    for value, buffers in (*runtime.buffer_plan.entry_inputs, *runtime.buffer_plan.entry_outputs):
        ty = fm.logical_type(module.node_map[value].type)
        tensor = torch.ones(tuple(d.fixed_value for d in ty.shape), dtype=getattr(torch, ty.dtype.value), device="cuda")
        for buffer in buffers:
            tensors[buffer] = tensor
    arguments = tuple(tensors[a["buffer"]] for a in runtime.external_arguments)
    runtime.prepare(*arguments)
    for _ in range(3):
        runtime.run_into(*arguments)
    torch.cuda.synchronize()
    for value, buffers in runtime.buffer_plan.entry_outputs:
        assert tensors[buffers[0]].shape == (0, 7)
    assert runtime.resource_report["spill_bytes"] == 0
