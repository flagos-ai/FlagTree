# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Execute a reusable pure function while all earlier results stay observable."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


@pytest.mark.parametrize("nested", [False, True])
def test_reusable_calls_preserve_borrowed_input_and_earlier_outputs(tmp_path, nested):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    ty = fm.tensor_type("bfloat16", (1, 128))

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            parameter = self.input("worker_parameter", ty, id="worker_parameter")
            doubled = fm.F.math.add(parameter, parameter, name="doubled")
            self.function("worker", (parameter,), (doubled,), attrs={"reusable": True, "noinline": True})
            callee = "worker"
            if nested:
                outer = self.input("outer_parameter", ty, id="outer_parameter")
                result = fm.F.builtin.call(outer, result_type=ty, callee="worker", name="outer_call")
                self.function("outer", (outer,), (result,), attrs={"reusable": True, "noinline": True})
                callee = "outer"
            value = self.input("value", ty, id="value")
            prepared = fm.F.math.add(value, value, name="prepared")
            first = fm.F.builtin.call(prepared, result_type=ty, callee=callee, name="first")
            second = fm.F.builtin.call(first, result_type=ty, callee=callee, name="second")
            self.function("main", (value,), (prepared, first, second))

    compiled = Compiler().compile(Graph().build()).module
    artifact = write_artifact(compiled, tmp_path / "borrowed", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    value = torch.arange(128, dtype=torch.float32, device="cuda").reshape(1, 128).bfloat16() / 16
    original = value.clone()
    outputs = [torch.empty_like(value) for _ in range(3)]
    bound = {buffer: value for _, buffers in runtime.buffer_plan.entry_inputs for buffer in buffers}
    for (_, buffers), output in zip(runtime.buffer_plan.entry_outputs, outputs, strict=True):
        assert len(buffers) == 1
        bound[buffers[0]] = output
    arguments = [bound[argument["buffer"]] for argument in runtime.external_arguments]
    runtime.prepare(*arguments)
    runtime.run_into(*arguments)
    torch.cuda.synchronize()
    assert runtime.resource_report["spill_bytes"] == 0
    torch.testing.assert_close(value, original, rtol=0, atol=0)
    for output, multiplier in zip(outputs, (2, 4, 8), strict=True):
        torch.testing.assert_close(output, original * multiplier, rtol=0, atol=0)
