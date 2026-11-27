# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Edited physical alignment must survive direct and nested source projection."""

import ast
from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton import describe_tir_package, render_tir_package


def _align_entry_allocations(module, alignment):
    physical = {}

    class Rebind(fm.TIRRewriter):
        def visit_buffer(self, value):
            buffer = value.mem_span.buffer
            if buffer.memory_space != "shared" or buffer.function != module.entry:
                return value
            assert buffer.offset % alignment == 0
            physical.setdefault(buffer.id, replace(buffer, alignment=alignment))
            return replace(value, mem_span=replace(value.mem_span, buffer=physical[buffer.id]))

    rewriter = Rebind()
    result = replace(module, execution_functions=tuple(rewriter.rewrite(value) for value in module.execution_functions))
    assert physical
    return result


@pytest.mark.parametrize("reusable, wrapper_depth, alignment", [
    (False, 0, 2048), (False, 0, 4096), (True, 0, 4096), (True, 2, 4096),
])
def test_stronger_physical_alignment_reaches_arena_after_python_resume(
    tmp_path, compile_pipeline_module, reusable, wrapper_depth, alignment,
):
    module = _align_entry_allocations(compile_pipeline_module(
        reusable=reusable, wrapper_depth=wrapper_depth), alignment)
    fm.verify_buffer_plan(module)
    path = tmp_path / "aligned.py"
    fm.emit_module(module, path)
    module = fm.load_module(path)
    package = describe_tir_package(module)
    schedule = package["pipeline_schedule"]
    assert schedule["shared_arena_alignment_bytes"] == alignment
    for stage in schedule["stages"]:
        for workspace in stage["workspaces"]:
            assert workspace["alignment_bytes"] == 1024
            assert workspace["allocation_alignment_bytes"] == alignment
    tree = ast.parse(render_tir_package(package, "unit"))
    arena = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                 and any(isinstance(value, ast.Name) and value.id == "_flagmega_shared_arena" for value in node.targets))
    assert next(ast.literal_eval(value.value) for value in arena.keywords if value.arg == "alignment_bytes") == alignment
