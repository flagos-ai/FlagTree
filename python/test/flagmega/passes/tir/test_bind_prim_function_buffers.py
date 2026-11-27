# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import IRVerificationError


def _add_module() -> fm.IRModule:
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value_type = fm.tensor_type("bfloat16", (1, 16))
            lhs = self.input("lhs", value_type, id="lhs")
            rhs = self.input("rhs", value_type, id="rhs")
            output = fm.F.math.add(lhs, rhs, name="output")
            self.function("main", (lhs, rhs), (output,))

    return Graph().build()


def test_bufferize_binds_formal_buffers_and_return_storage():
    selected = Compiler().compile(_add_module(), stop_after="lower-tir").module
    assert all(not value.buffers for value in selected.kernel_definitions[0].parameters)

    allocated = Compiler().compile(selected, stop_after="bufferize").module
    function = next(
        value
        for value in allocated.kernel_definitions
        if fm.kernel_dispatch_of(value).semantic_op == "math.vectorized_binary"
    )
    lhs, rhs, output = function.parameters

    assert [value.role for value in function.parameters] == [
        fm.PrimParameterRole.INPUT,
        fm.PrimParameterRole.INPUT,
        fm.PrimParameterRole.OUTPUT,
    ]
    assert lhs.buffers[0].name == "lhs"
    assert rhs.buffers[0].mem_span.buffer.id != lhs.buffers[0].mem_span.buffer.id
    assert output.buffers[0].mem_span.size.fixed_value == 32
    assert isinstance(function.results.values[0].value, fm.BufferTuple)
    assert function.results.values[0].value.buffers == output.buffers


def test_formal_buffer_python_checkpoint_round_trips(tmp_path):
    allocated = Compiler().compile(_add_module(), stop_after="bufferize").module

    loaded = fm.load_module(fm.emit_module(allocated, tmp_path / "allocated.py"))

    assert loaded.semantic_hash == allocated.semantic_hash
    assert loaded.kernel_definitions[0].parameters[-1].buffers


def test_prim_function_verifier_rejects_wrong_formal_role():
    allocated = Compiler().compile(_add_module(), stop_after="bufferize").module
    function = allocated.kernel_definitions[0]
    parameter = function.parameters[0]
    buffer = parameter.buffers[0]
    wrong_physical = replace(buffer.mem_span.buffer, memory_space="output")
    wrong_buffer = replace(buffer, mem_span=fm.MemSpan(wrong_physical))
    wrong_parameter = replace(parameter, buffers=(wrong_buffer,))
    wrong_function = replace(
        function,
        parameters=(wrong_parameter, *function.parameters[1:]),
    )

    with pytest.raises(IRVerificationError, match="wrong ABI role"):
        fm.verify_module(replace(allocated, kernel_definitions=(wrong_function,)))


def test_buffer_plan_verifier_matches_serialized_buffers_to_graph_value_types():
    allocated = Compiler().compile(_add_module(), stop_after="bufferize").module
    plan = {
        **allocated.metadata["buffer_plan"],
        "buffers": [dict(value) for value in allocated.metadata["buffer_plan"]["buffers"]],
    }
    lhs = next(value for value in plan["buffers"] if value["id"] == "lhs")
    lhs["dtype"] = "float32"
    corrupted = replace(allocated, metadata={**allocated.metadata, "buffer_plan": plan})

    with pytest.raises(IRVerificationError, match="does not match its logical IR type"):
        fm.verify_buffer_plan(corrupted)
