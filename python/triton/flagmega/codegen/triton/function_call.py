# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Lower nested FunctionBufferPlan edges to Triton call expressions."""

from __future__ import annotations

from collections.abc import Mapping

from triton.flagmega.codegen.triton.physical_access import (
    emit_scalar_immediate,
    emit_storage_pointer,
)
from triton.flagmega.codegen.triton.pool_abi import emit_pool_scope_base
from triton.flagmega.codegen.triton.runtime_binding import (
    describe_function_runtime_binding,
)
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import IRModule


def emit_function_call_arguments(
    module: IRModule,
    caller_binding: Mapping[str, object],
    call_event: Mapping[str, object],
) -> tuple[str, ...]:
    """Emit actuals in the callee's verified flattened signature order."""

    if call_event.get("kind") != "function_call":
        raise CodegenError("Triton function-call lowering requires a function event.")
    callee = str(call_event["callee"])
    callee_binding = describe_function_runtime_binding(
        module,
        function_name=callee,
    )
    edges = {
        str(edge["formal"]): edge
        for edge in (*call_event["arguments"], *call_event["results"])
    }
    expressions: dict[str, str] = {}
    for argument in callee_binding["arguments"]:
        formal = str(argument["buffer"])
        try:
            edge = edges[formal]
        except KeyError as error:
            raise CodegenError(
                f"Call {call_event['call']!r} has no actual buffer for "
                f"@{callee} formal {formal!r}."
            ) from error
        actual = str(edge["actual_runtime_argument"])
        if edge["actual_runtime_value_kind"] == "immediate":
            expression = emit_scalar_immediate(edge["formal_abi"], actual)
        elif edge["actual_runtime_value_kind"] == "scalar":
            expression = actual
        else:
            expression = emit_storage_pointer(edge["actual_abi"], actual)
        expressions[str(argument["name"])] = expression

    caller_pools = {
        str(pool["storage"]): pool
        for pool in caller_binding["pools"]
    }
    frames = {
        str(frame["memory_space"]): frame
        for frame in call_event["memory_pools"]
    }
    for pool in callee_binding["pools"]:
        storage = str(pool["storage"])
        try:
            caller_pool = caller_pools[storage]
        except KeyError as error:
            raise CodegenError(
                f"Call {call_event['call']!r} requires callee {storage!r} "
                "storage, but the caller has no such runtime pool."
            ) from error
        caller_pool_name = str(caller_pool["name"])
        frame = frames.get(storage)
        if frame is not None:
            offset = int(frame["offset"])
            scope_base = emit_pool_scope_base(caller_pool, caller_pool_name)
            expression = (
                scope_base if offset == 0 else f"({scope_base} + {offset})"
            )
        else:
            expression = caller_pool_name
        expressions[str(pool["name"])] = expression

    try:
        return tuple(
            expressions[str(name)] for name in callee_binding["signature"]
        )
    except KeyError as error:
        raise CodegenError(
            f"Call {call_event['call']!r} did not bind callee signature "
            f"argument {error.args[0]!r}."
        ) from error


__all__ = ["emit_function_call_arguments"]
