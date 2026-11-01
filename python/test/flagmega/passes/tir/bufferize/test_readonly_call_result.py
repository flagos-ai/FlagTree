# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A callee's immutable result aliases its loaded readonly allocation."""

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.ir.bufferization import BufferPlan


def graph(*, nested=False):
    class Graph(fm.Module):
        def forward(self):
            kind = fm.tensor_type("float32", (128,))
            source = self.input("source", kind, id="source")
            parameter = self.input("parameter", kind, id="parameter")
            one = fm.F.builtin.splat_const(kind, 1., name="one")
            doubled = fm.F.math.add(parameter, parameter, name="doubled")
            self.function("worker", (parameter,), (doubled, one), attrs={"reusable": True})
            callee = "worker"
            if nested:
                outer = self.input("outer", kind, id="outer")
                inner = fm.F.builtin.call(outer, callee="worker", result_type=fm.TupleType((kind, kind)), name="inner")
                inner_value, inner_constant = fm.F.tensors.get_items(inner, 0, 1)
                self.function("outer_worker", (outer,), (inner_value, inner_constant), attrs={"reusable": True})
                callee = "outer_worker"
            call = fm.F.builtin.call(source, callee=callee, result_type=fm.TupleType((kind, kind)), name="call")
            value, constant = fm.F.tensors.get_items(call, 0, 1)
            output = fm.F.math.add(value, constant, name="result")
            self.function("main", (source,), (output,))
    return Graph(dialect="high_level", stage="imported", entry="main").build()


@pytest.mark.parametrize("level", ["fast", "optimized"])
@pytest.mark.parametrize("nested", [False, True])
def test_readonly_result_is_bound_to_rdata(level, nested):
    compiled = Compiler(CompileOptions(bufferize_opt_level=level)).compile(graph(nested=nested)).module
    plan = BufferPlan.from_data(compiled.metadata["buffer_plan"])
    readonly_results = [(formal, actual) for function in plan.functions for call in function.calls
                        for formal, actual in call.results if plan.buffer_map[formal].storage == "rdata"]
    assert readonly_results, "Exercise the immutable function result."
    for formal, actual in readonly_results:
        assert plan.buffer_map[actual].storage == "rdata"
        assert plan.buffer_map[formal].mem_span.must_alias(plan.buffer_map[actual].mem_span)


@pytest.mark.parametrize("level", ["fast", "optimized"])
@pytest.mark.parametrize("nested", [False, True])
def test_readonly_result_executes_with_real_constant_bytes(tmp_path, level, nested):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime
    compiled = Compiler(CompileOptions(bufferize_opt_level=level)).compile(graph(nested=nested)).module
    class Checkpoint:
        def load_tensor(self, key, *, device="cpu"):
            raise AssertionError(f"A literal result requires no checkpoint weight: {key}")
    runtime = load(write_artifact(compiled, tmp_path / "artifact", checkpoint=Checkpoint(),
                                 target="nvidia-sm90", emit_executable=True), device="cuda:0")
    try:
        values = {spec["buffer"]: torch.zeros(128, device="cuda") for spec in runtime.external_arguments}
        args = tuple(values[spec["buffer"]] for spec in runtime.external_arguments)
        Runtime.prepare(runtime, *args)
        source, = dict(runtime.buffer_plan.entry_inputs)["source"]
        output, = runtime.buffer_plan.entry_outputs[0][1]
        for mutation in range(3):
            values[source].copy_(torch.arange(128, device="cuda") / 8 + mutation / 4)
            values[output].fill_(float("nan"))
            Runtime.run_into(runtime, *args)
            torch.cuda.synchronize()
            torch.testing.assert_close(values[output], values[source] * 2 + 1, rtol=0, atol=0)
    finally:
        runtime.close()
