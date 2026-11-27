# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Execution boundaries are semantic, not a local/collective heuristic."""

import pytest

from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry


@pytest.mark.parametrize("execution_kind", [
    "local_shard", "synchronized_local", "collective",
])
def test_op_without_internal_grid_barrier_has_device_function_boundary(execution_kind):
    template = TritonTemplateRegistry().environment.from_string(
        '{% import "entrypoints/_common.py.jinja" as entry with context %}'
        '{{ entry.kernel_decorator(call) }}'
    )
    source = template.render(call={
        "execution_kind": execution_kind,
        "internal_grid_barriers": 0,
    })
    assert source.strip() == "@triton.jit(noinline=True)"


@pytest.mark.parametrize("execution_kind", ["synchronized_local", "collective"])
def test_internal_grid_barrier_schedule_retains_entry_scratch_scope(execution_kind):
    # Until these schedules are phase-split, a grid barrier cannot move into a
    # Triton device function: its implicit scratch ABI is entry-relative.
    template = TritonTemplateRegistry().environment.from_string(
        '{% import "entrypoints/_common.py.jinja" as entry with context %}'
        '{{ entry.kernel_decorator(call) }}'
    )
    source = template.render(call={
        "execution_kind": execution_kind,
        "internal_grid_barriers": 1,
    })
    assert source.strip() == "@triton.jit"
