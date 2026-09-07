# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.ir.bufferization import (
    MemoryRange,
    MemorySynchronizationPlan,
    SynchronizationEvent,
)
from triton.flagmega.passes.tir import (
    materialize_memory_synchronization,
    memory_synchronization_from_execution_functions,
)

from ..execution.helpers import scheduled_nested_module


def test_axis_group_barrier_survives_python_dump_edit_resume(tmp_path):
    plan = MemorySynchronizationPlan((SynchronizationEvent(
        "main",
        "prepared",
        "nested",
        "grid",
        ("WRITE->READ",),
        (MemoryRange("workspace", "workspace:@main", 0, 32, "read"),),
        (0,),
    ),))
    module = materialize_memory_synchronization(
        scheduled_nested_module(), plan
    )

    path = fm.emit_module(module, tmp_path / "synchronized.py")
    source = path.read_text(encoding="utf-8")
    resumed = fm.load_module(path)

    assert "axis_group_axes=(" in source
    assert "T.barrier(" in source
    assert memory_synchronization_from_execution_functions(resumed) == plan
    assert resumed.semantic_hash == module.semantic_hash
