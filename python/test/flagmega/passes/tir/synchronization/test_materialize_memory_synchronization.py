# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
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


def _plan():
    return MemorySynchronizationPlan((SynchronizationEvent(
        "main",
        "prepared",
        "nested",
        "grid",
        ("WRITE->READ",),
        (MemoryRange(
            "workspace",
            "workspace:@main",
            0,
            32,
            "read",
        ),),
    ),))


def test_plan_is_materialized_immediately_before_dependent_call():
    result = materialize_memory_synchronization(
        scheduled_nested_module(), _plan()
    )
    fields = result.execution_function_map["main"].body.fields

    assert [type(value) for value in fields] == [
        fm.KernelInvoke,
        fm.Barrier,
        fm.PrimFunctionCall,
        fm.KernelInvoke,
    ]
    barrier = fields[1]
    assert barrier.scope is fm.BarrierScope.CHIP
    assert barrier.after == ("prepared",)
    assert barrier.before == "nested"
    assert barrier.ranges[0] == fm.T.synchronization_range(
        "workspace", "workspace:@main", 0, 32, "read"
    )


def test_materialized_barriers_reconstruct_the_exact_diagnostic_plan():
    result = materialize_memory_synchronization(
        scheduled_nested_module(), _plan()
    )

    assert memory_synchronization_from_execution_functions(result) == _plan()


def test_plan_with_non_topological_predecessor_fails_fast():
    invalid = MemorySynchronizationPlan((SynchronizationEvent(
        "main", "output", "nested", "grid", (), ()
    ),))

    try:
        materialize_memory_synchronization(
            scheduled_nested_module(), invalid
        )
    except IRVerificationError as error:
        assert "not topologically ordered" in str(error)
    else:
        raise AssertionError("a barrier predecessor must execute first")
