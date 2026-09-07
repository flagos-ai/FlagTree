# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Generic function reachability and call-graph traversal analysis."""

from __future__ import annotations

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import Function, IRModule, Node


def function_nodes(module: IRModule, function: Function) -> tuple[Node, ...]:
    node_map = module.node_map
    reachable = set(function.parameters)
    pending = list(function.outputs)
    while pending:
        node_id = pending.pop()
        if node_id in reachable:
            continue
        try:
            node = node_map[node_id]
        except KeyError as error:
            raise IRVerificationError(
                f"Function @{function.name} references missing node {node_id!r}.",
                stage=module.stage,
            ) from error
        reachable.add(node_id)
        pending.extend(node.inputs)
    return tuple(node for node in module.nodes if node.id in reachable)


def callee_first_functions(module: IRModule) -> tuple[Function, ...]:
    functions = module.function_map
    callees: dict[str, tuple[str, ...]] = {}
    for function in module.functions:
        values = []
        for node in function_nodes(module, function):
            if node.op not in {"builtin.call", "tir.call"}:
                continue
            callee = str(node.attrs.get("callee", ""))
            if callee in module.kernel_callable_map:
                # Backend PrimFunctions are leaves in the graph-function call
                # graph. They have their own pass traversal and ABI verifier.
                continue
            if callee not in functions:
                raise IRVerificationError(
                    f"Call {node.id!r} in @{function.name} references unknown @{callee}.",
                    stage=module.stage,
                    node_id=node.id,
                )
            values.append(callee)
        callees[function.name] = tuple(dict.fromkeys(values))

    result: list[Function] = []
    state: dict[str, int] = {}

    def visit(name: str, stack: tuple[str, ...]) -> None:
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            cycle = " -> ".join((*stack, name))
            raise IRVerificationError(
                f"Callee-first traversal does not support recursive call graph: {cycle}."
            )
        state[name] = 1
        for callee in callees[name]:
            visit(callee, (*stack, name))
        state[name] = 2
        result.append(functions[name])

    for function in module.functions:
        visit(function.name, ())
    return tuple(result)


def reusable_function_node_ids(module: IRModule) -> frozenset[str]:
    """Return nodes owned by explicitly reusable graph functions."""

    return frozenset(
        node.id
        for function in module.functions
        if function.attrs.get("reusable") is True
        for node in function_nodes(module, function)
    )


def static_function_invocation_counts(module: IRModule) -> dict[str, int]:
    """Count graph-function invocations from the entry without inlining."""

    function_map = module.function_map
    if module.entry not in function_map:
        raise IRVerificationError(
            f"Entry function @{module.entry} is not defined.", stage=module.stage)
    direct: dict[str, dict[str, int]] = {}
    for function in module.functions:
        counts: dict[str, int] = {}
        for node in function_nodes(module, function):
            if node.op not in {"builtin.call", "tir.call"}:
                continue
            callee = str(node.attrs.get("callee", ""))
            if callee in module.kernel_callable_map:
                continue
            if callee not in function_map:
                raise IRVerificationError(
                    f"Call {node.id!r} in @{function.name} references unknown "
                    f"graph function @{callee}.",
                    stage=module.stage,
                    node_id=node.id,
                )
            counts[callee] = counts.get(callee, 0) + 1
        direct[function.name] = counts

    order: list[str] = []
    state: dict[str, int] = {}

    def visit(name: str, path: tuple[str, ...]) -> None:
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            cycle = " -> ".join((*path, name))
            raise IRVerificationError(
                "Static invocation analysis does not support recursive "
                f"function graph: {cycle}.",
                stage=module.stage,
            )
        state[name] = 1
        for callee in direct[name]:
            visit(callee, (*path, name))
        state[name] = 2
        order.append(name)

    visit(module.entry, ())
    maximum = (1 << 63) - 1
    invocations = {module.entry: 1}
    for caller in reversed(order):
        caller_count = invocations.get(caller)
        if caller_count is None:
            continue
        for callee, calls_per_invocation in direct[caller].items():
            updated = (
                invocations.get(callee, 0)
                + caller_count * calls_per_invocation
            )
            if updated > maximum:
                raise IRVerificationError(
                    f"Static invocation count for @{callee} exceeds signed "
                    "64-bit range.",
                    stage=module.stage,
                )
            invocations[callee] = updated
    return invocations


def static_node_invocation_counts(module: IRModule) -> dict[str, int]:
    """Map each reachable expression to its static execution multiplicity."""

    function_counts = static_function_invocation_counts(module)
    result: dict[str, int] = {}
    for function_name, invocation_count in function_counts.items():
        for node in function_nodes(module, module.function_map[function_name]):
            result[node.id] = result.get(node.id, 0) + invocation_count
    return result


__all__ = [
    "callee_first_functions",
    "function_nodes",
    "reusable_function_node_ids",
    "static_function_invocation_counts",
    "static_node_invocation_counts",
]
