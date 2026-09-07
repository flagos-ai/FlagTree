# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Materialize a physical hazard plan as executable TIR barriers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import (
    Barrier,
    BarrierScope,
    IRModule,
    PrimFunctionCall,
    KernelInvoke,
    Sequential,
    SynchronizationRange,
    execution_items_of,
)
from triton.flagmega.ir.bufferization import (
    MemoryRange,
    MemorySynchronizationPlan,
    SynchronizationEvent,
)


def materialize_memory_synchronization(
    module: IRModule,
    plan: MemorySynchronizationPlan,
) -> IRModule:
    """Insert every planned barrier immediately before its dependent call."""

    by_function: dict[str, dict[str, list[SynchronizationEvent]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for event in plan.events:
        by_function[event.function][event.before].append(event)

    functions = []
    matched: set[tuple[str, str, str]] = set()
    for function in module.execution_functions:
        fields = []
        seen_calls: set[str] = set()
        for statement in function.body.fields:
            if not isinstance(statement, (PrimFunctionCall, KernelInvoke)):
                raise IRVerificationError(
                    f"ExecutionFunction @{function.name} must be a flat call "
                    "sequence before PlanMemorySynchronization."
                )
            for event in by_function.get(function.name, {}).get(
                statement.call_id, ()
            ):
                if event.after not in seen_calls:
                    raise IRVerificationError(
                        f"Synchronization {event.after!r}->{event.before!r} in "
                        f"@{function.name} is not topologically ordered."
                    )
                fields.append(_barrier(event))
                matched.add((event.function, event.after, event.before))
            fields.append(statement)
            seen_calls.add(statement.call_id)
        functions.append(replace(function, body=Sequential(tuple(fields))))

    expected = {
        (event.function, event.after, event.before) for event in plan.events
    }
    if matched != expected:
        raise IRVerificationError(
            "Memory synchronization plan references calls absent from the "
            f"execution IR: {sorted(expected - matched)}."
        )
    return replace(module, execution_functions=tuple(functions))


def memory_synchronization_from_execution_functions(
    module: IRModule,
) -> MemorySynchronizationPlan:
    events = []
    for function in module.execution_functions:
        pending: list[Barrier] = []
        for item in execution_items_of(function):
            if isinstance(item, Barrier):
                pending.append(item)
                continue
            if not isinstance(item, (PrimFunctionCall, KernelInvoke)):
                continue
            for barrier in pending:
                if barrier.before != item.call_id:
                    raise IRVerificationError(
                        f"Barrier before {barrier.before!r} is placed before "
                        f"execution call {item.call_id!r} in @{function.name}."
                    )
                events.append(SynchronizationEvent(
                    function.name,
                    barrier.after[-1],
                    barrier.before,
                    "grid" if barrier.scope is BarrierScope.CHIP else "block",
                    barrier.hazards,
                    tuple(MemoryRange(
                        value.storage,
                        value.physical_id,
                        value.offset,
                        value.nbytes,
                        value.mode,
                    ) for value in barrier.ranges),
                    barrier.axis_group_axes,
                ))
            pending.clear()
        if pending:
            raise IRVerificationError(
                f"ExecutionFunction @{function.name} ends with an unattached barrier."
            )
    return MemorySynchronizationPlan(tuple(events))


def _barrier(event: SynchronizationEvent) -> Barrier:
    if event.scope in {"grid", "chip"}:
        scope = BarrierScope.CHIP
    elif event.scope == "block":
        scope = BarrierScope.BLOCK
    else:
        raise IRVerificationError(
            f"Unknown memory synchronization scope {event.scope!r}."
        )
    return Barrier(
        scope,
        (event.after,),
        event.before,
        hazards=event.hazards,
        ranges=tuple(SynchronizationRange(
            value.storage,
            value.physical_id,
            value.offset,
            value.nbytes,
            value.access,
        ) for value in event.ranges),
        axis_group_axes=event.axis_group_axes,
    )


__all__ = [
    "materialize_memory_synchronization",
    "memory_synchronization_from_execution_functions",
]
