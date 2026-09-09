# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Real artifacts bind offline-derived weights to a reused device function."""

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime import load
from .helpers import module


@pytest.mark.parametrize("level", ["fast", "optimized"])
@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_offline_weights_and_reused_function_execute_exactly(tmp_path, level, distributed, nested):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    source = module(distributed=distributed, nested=nested)
    weights = {f"weight{i}": (torch.arange(8, dtype=torch.float32) * .017 + i).bfloat16() for i in range(2)}

    class Checkpoint:

        def load_tensor(self, key, *, device="cpu"):
            return weights[key].to(device)

    compiled = Compiler(CompileOptions(bufferize_opt_level=level)).compile(source).module
    artifact = write_artifact(compiled, tmp_path / "artifact", checkpoint=Checkpoint(), target="nvidia-sm90",
                              emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    tensors = {}
    for name, buffers in (*runtime.buffer_plan.entry_inputs, *runtime.buffer_plan.entry_outputs):
        value_type = fm.logical_type(compiled.node_map[name].type)
        tensor = torch.empty(tuple(d.fixed_value for d in value_type.shape),
                             dtype=getattr(torch, value_type.dtype.value), device="cuda")
        assert len(buffers) == 1
        descriptor = runtime.buffer_plan.buffer_map[buffers[0]]
        assert descriptor.physical_access_span.nbytes <= tensor.numel() * tensor.element_size()
        tensors[buffers[0]] = tensor
    actuals = tuple(tensors[a["buffer"]] for a in runtime.external_arguments)
    runtime.prepare(*actuals)
    input_buffer, = dict(runtime.buffer_plan.entry_inputs)["runtime"]
    for repeat in range(3):
        values = torch.linspace(-2, 2, 8, device="cuda") + repeat * .125
        tensors[input_buffer].copy_(values)
        for _, buffers in runtime.buffer_plan.entry_outputs:
            tensors[buffers[0]].fill_(float("nan"))
        runtime.run_into(*actuals)
        torch.cuda.synchronize()
        for i, (_, buffers) in enumerate(runtime.buffer_plan.entry_outputs):
            expected = values + weights[f"weight{i}"].float().cuda() * 2
            assert torch.equal(tensors[buffers[0]].view(torch.int32), expected.view(torch.int32))
    assert runtime.resource_report["spill_bytes"] == 0
