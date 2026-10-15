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
    KernelInvoke,
    PrimFunctionCall,
    Sequential,
    SynchronizationRange,
    execution_items_of,
)
from triton.flagmega.ir.bufferization import (
    MemoryRange,
    MemorySynchronizationPlan,
    SynchronizationEvent,
)
from triton.flagmega.ir.tir.for_loop import For


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
    repeated = _repeated_execution_functions(module)
    matched: set[tuple[str, str, str]] = set()
    for function in module.execution_functions:
        fields = []
        seen_calls: set[str] = set()
        # A repeated function's own trailing access precedes its leading one
        # on the next execution, so that hazard's producer is lexically later
        # than its consumer.  Program order alone cannot discharge it; the
        # barrier is still emitted before the consumer, where every execution
        # reaches it.  Functions that run once keep the strict topological
        # requirement.
        body_calls = {
            statement.call_id
            for statement in function.body.fields
            if isinstance(statement, (PrimFunctionCall, KernelInvoke))
        } if function.name in repeated else frozenset()
        for statement in function.body.fields:
            if isinstance(statement, For):
                fields.append(statement)
                continue
            if not isinstance(statement, (PrimFunctionCall, KernelInvoke)):
                raise IRVerificationError(
                    f"ExecutionFunction @{function.name} must be a flat call "
                    "sequence before PlanMemorySynchronization."
                )
            for event in by_function.get(function.name, {}).get(
                statement.call_id, ()
            ):
                if event.after not in seen_calls and event.after not in body_calls:
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


def _repeated_execution_functions(module: IRModule) -> frozenset[str]:
    """Name every execution function that runs more than once per launch.

    A structured loop's body repeats, and so does anything it calls, because
    the body invokes that callee on every trip.
    """

    from triton.flagmega.ir import execution_calls_of

    result = {
        function.name
        for function in module.execution_functions
        if function.attrs.get("region_kind") == "for"
    }
    names = {function.name for function in module.execution_functions}

    def visit(node) -> None:
        if isinstance(node, For) and node.body_function in names:
            result.add(node.body_function)
        for child in getattr(node, "fields", ()) or ():
            visit(child)
        body = getattr(node, "body", None)
        if body is not None and body is not node:
            visit(body)

    for function in module.execution_functions:
        visit(function.body)
    by_name = {value.name: value for value in module.execution_functions}
    pending = list(result)
    while pending:
        current = by_name.get(pending.pop())
        if current is None:
            continue
        for call in execution_calls_of(current):
            if call.callee in names and call.callee not in result:
                result.add(call.callee)
                pending.append(call.callee)
    return frozenset(result)


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
                    {
                        BarrierScope.DEVICE: "device",
                        BarrierScope.CHIP: "grid",
                    }.get(barrier.scope, "block"),
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
    if event.scope == "device":
        scope = BarrierScope.DEVICE
    elif event.scope in {"grid", "chip"}:
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
