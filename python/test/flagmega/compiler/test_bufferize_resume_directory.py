# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Changing allocation strategy resumes the whole pre-allocation call graph."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.diagnostics import DumpFlags
from triton.flagmega.options import CompileOptions


def reusable_graph():

    class Graph(fm.Module):

        def forward(self):
            tensor = fm.tensor_type("bfloat16", (1, 64))
            parameter = self.input("parameter", tensor)
            self.function("worker", (parameter, ), (fm.F.math.silu(parameter), ),
                          attrs={"reusable": True, "noinline": True})
            value = self.input("value", tensor)
            first = fm.F.builtin.call(value, callee="worker", result_type=tensor, name="first")
            second = fm.F.builtin.call(first, callee="worker", result_type=tensor, name="second")
            self.function("main", (value, ), (second, ))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


@pytest.mark.parametrize("initial,final", [("fast", "optimized"), ("optimized", "fast")])
def test_change_level_resumes_complete_before_directory(tmp_path, monkeypatch, initial, final):
    from triton.flagmega.passes.auto_distributed import AutoDistributedPass
    from triton.flagmega.passes.tir.bufferize import FirstFitBufferAllocator, SATBufferAllocator

    dumps = tmp_path / "dumps"
    original = Compiler(CompileOptions(bufferize_opt_level=initial, work_dir=dumps,
                                       dump_flags=DumpFlags.PASS_IR)).compile(reusable_graph()).module
    assert fm.verify_buffer_plan(original).optimization_level == initial
    before_directories = tuple(dumps.glob("**/*_Bufferize/Before"))
    assert len(before_directories) == 1
    before_directory, = before_directories
    before = fm.load_module(before_directory)
    assert "buffer_plan" not in before.metadata
    assert {path.stem for path in before_directory.glob("*.py")} == {fn.name for fn in before.functions}
    assert {"main", "worker"} <= {fn.name for fn in before.functions}
    input_hash = before.semantic_hash

    def unexpected_distribution(*args, **kwargs):
        pytest.fail("Changing allocator at Bufferize/Before must not redo AutoDist")

    monkeypatch.setattr(AutoDistributedPass, "_build_graph", unexpected_distribution)
    selected = FirstFitBufferAllocator if final == "fast" else SATBufferAllocator
    allocate = selected.allocate
    allocations = []

    def record_allocation(self, *args, **kwargs):
        allocations.append(args)
        return allocate(self, *args, **kwargs)

    monkeypatch.setattr(selected, "allocate", record_allocation)
    resumed = Compiler(CompileOptions(bufferize_opt_level=final)).compile(before).module
    plan = fm.verify_buffer_plan(resumed)
    assert allocations, "Resume must compute a new placement, not relabel the saved plan"
    assert plan.optimization_level == final
    assert plan.allocator == selected.name
    assert plan.allocation_records and all(record.allocator == selected.name for record in plan.allocation_records)
    assert before.semantic_hash == input_hash
    assert fm.load_module(before_directory) == before
    assert fm.load_module(fm.emit_module(resumed, tmp_path / "resumed.py")) == resumed
    # The repeated worker remains a reusable graph function after either plan.
    calls = [node for node in resumed.nodes if node.op == "tir.call" and node.attrs.get("callee") == "worker"]
    assert len(calls) == 2
    assert any(first.id in second.inputs for first in calls for second in calls if first is not second)
