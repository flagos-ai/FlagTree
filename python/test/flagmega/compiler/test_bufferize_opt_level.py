# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import IRVerificationError, StageError
from triton.flagmega.ir.printer import script_source
from triton.flagmega.options import CompileOptions
from triton.flagmega.targets import get_target


def graph():

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1, 64)))
            self.function("main", (value, ), (fm.F.math.silu(value), ))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def test_compiler_level_does_not_mutate_registered_target():
    target = get_target("nvidia-sm90")
    policy = target.bufferization_policy
    fast = Compiler(CompileOptions(bufferize_opt_level="fast"))
    optimized = Compiler(CompileOptions(bufferize_opt_level="optimized"))
    assert target.bufferization_policy is policy
    assert policy.options.optimization_level == "optimized"
    assert fast.target.bufferization_policy.options.optimization_level == "fast"
    assert optimized.target.bufferization_policy.options.optimization_level == "optimized"
    with pytest.raises(ValueError, match="bufferize_opt_level"):
        CompileOptions(bufferize_opt_level="typo")


def test_fast_compilation_never_invokes_sat_allocator(monkeypatch):
    from triton.flagmega.passes.tir.bufferize import FirstFitBufferAllocator, SATBufferAllocator

    def unexpected_sat(*args, **kwargs):
        pytest.fail("Fast bufferization must not invoke the SAT allocator.")

    monkeypatch.setattr(SATBufferAllocator, "allocate", unexpected_sat)
    result = Compiler(CompileOptions(bufferize_opt_level="fast")).compile(graph()).module
    plan = fm.verify_buffer_plan(result)
    assert plan.optimization_level == "fast"
    assert plan.allocator == FirstFitBufferAllocator.name
    assert plan.allocation_records
    assert all(record.allocator == FirstFitBufferAllocator.name for record in plan.allocation_records)


@pytest.mark.parametrize("level", ["fast", "optimized"])
def test_level_roundtrip_and_resume_requires_pre_bufferize_boundary(tmp_path, level):
    original = Compiler().compile(graph(), stop_after="plan-function-memory").module
    result = Compiler(CompileOptions(bufferize_opt_level=level)).compile(original).module
    plan = fm.verify_buffer_plan(result)
    assert plan.optimization_level == level
    assert plan.allocation_records
    readable = script_source(result)
    assert f"buffer allocation: {level} ({plan.allocator})" in readable
    assert ("OPTIMAL" if level == "optimized" else "FEASIBLE") in readable
    assert fm.load_module(fm.emit_module(result, tmp_path / "final.py")) == result
    assert Compiler().compile(result).module == result
    assert Compiler(CompileOptions(bufferize_opt_level=level)).compile(result).module == result
    other = "optimized" if level == "fast" else "fast"
    compiler = Compiler(CompileOptions(bufferize_opt_level=other))
    with pytest.raises(StageError, match="before Bufferize"):
        compiler.compile(result)
    with pytest.raises(StageError, match="before Bufferize"):
        compiler.run_stage(result, "bufferize")
    from dataclasses import replace
    mismatched = replace(
        result,
        metadata={**result.metadata, "buffer_plan": {**result.metadata["buffer_plan"], "optimization_level": other}})
    with pytest.raises(IRVerificationError, match="allocator"):
        fm.verify_buffer_plan(mismatched)
