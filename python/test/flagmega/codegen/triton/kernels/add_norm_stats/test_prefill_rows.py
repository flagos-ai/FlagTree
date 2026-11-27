# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Prefill residual fusion must retain a separate statistic for every token."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule


@pytest.mark.parametrize("rows", [3, 32])
def test_prefill_residual_statistics(tmp_path, rows):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")

    class Graph(fm.Module):
        def forward(self):
            a = self.input("a", fm.tensor_type("float32", (rows, 512)), id="a")
            b = self.input("b", fm.tensor_type("bfloat16", (rows, 512)), id="b")
            value = fm.F.math.add(a, fm.F.tensors.cast(b, "float32"), name="residual")
            stats = fm.F.nn.norm_stats(value, axis=-1, use_mean=False, name="stats")
            self.function("main", (a, b), (value, stats))

    module = Compiler().compile(Graph(dialect="high_level", stage="imported", entry="main").build()).module
    runtime = load(write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True),
                   device="cuda:0")
    try:
        a = (torch.arange(rows * 512, device="cuda") % 11 / 8).reshape(rows, 512).float()
        b = (torch.arange(rows * 512, device="cuda") % 7 / 4).reshape(rows, 512).bfloat16()
        values = {}
        for spec in runtime.external_arguments:
            buffer = runtime.buffer_plan.buffer_map[spec["buffer"]]
            lanes = getattr(buffer.dtype, "lanes", ())
            dtype = buffer.dtype.elem_type if lanes else buffer.dtype
            shape = (*buffer.shape, *lanes)
            values[buffer.id] = (torch.empty(shape, device="cuda", dtype=getattr(torch, dtype.value))
                                 if spec["role"] == "result" else {"a": a, "b": b}[spec["value"]].reshape(shape))
        arguments = tuple(values[spec["buffer"]] for spec in runtime.external_arguments)
        output, stats = (values[item[1][0]] for item in runtime.buffer_plan.function_map["main"].outputs)
        GeneratedTirCallGraphModule.prepare(runtime, *arguments)
        for _ in range(2):
            a.add_(.125)
            expected = a + b.float()
            GeneratedTirCallGraphModule.run_into(runtime, *arguments)
            torch.cuda.synchronize()
            torch.testing.assert_close(output.reshape_as(expected), expected, rtol=0, atol=0)
            torch.testing.assert_close(stats.reshape(rows), expected.square().sum(-1), rtol=0, atol=0)
    finally:
        runtime.close()
