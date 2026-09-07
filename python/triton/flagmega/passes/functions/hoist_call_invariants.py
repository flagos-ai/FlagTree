# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Share pure parameter-dependent expressions between static function calls.

Only identical SSA arguments prove invariance; equal types, names, and metadata
do not. Tensor values are immutable, while Ref/pointer state and effectful reads
are never candidates. Each caller gets its own computation, before its first
use, and the callee remains a single reusable function with a refined ABI.
"""

from __future__ import annotations

from dataclasses import replace

from triton.flagmega.ir import IRModule, Node, PointerType, TensorType, TupleType
from .graph import callee_first_functions, function_nodes


def _immutable(value_type) -> bool:
    return (isinstance(value_type, TensorType) and not isinstance(value_type.dtype, PointerType)) or (
        isinstance(value_type, TupleType)
        and all(_immutable(field) for field in value_type.fields)
    )


def hoist_call_invariant_expressions(module: IRModule) -> IRModule:
    for function in callee_first_functions(module):
        if function.name != module.entry:
            module = _hoist_function(module, function.name)
    return module


def _hoist_function(module: IRModule, name: str) -> IRModule:
    function = module.function_map[name]
    bodies = {f.name: function_nodes(module, f) for f in module.functions}
    foreign_ids = {node.id for owner, body in bodies.items() if owner != name for node in body}
    calls_by_caller = {
        owner: tuple(n for n in body if n.op == "builtin.call" and n.attrs["callee"] == name)
        for owner, body in bodies.items()
    }
    calls_by_caller = {owner: calls for owner, calls in calls_by_caller.items() if calls}
    if not any(len(calls) > 1 for calls in calls_by_caller.values()):
        return module
    nodes = module.node_map
    invariant = {
        parameter for index, parameter in enumerate(function.parameters)
        if _immutable(nodes[parameter].type)
        and all(len({call.inputs[index] for call in calls}) == 1 for calls in calls_by_caller.values())
    }
    if not invariant:
        return module

    # Empty dependencies denote unchanged constants, not expressions to hoist.
    dependencies = {parameter: frozenset({parameter}) for parameter in invariant}
    movable: list[Node] = []
    for node in bodies[name]:
        if node.id in function.parameters:
            continue
        if (not node.effect.is_pure or not _immutable(node.type)
                or node.op in {"builtin.var", "builtin.call", "tir.call"}
                or any(value not in dependencies for value in node.inputs)):
            continue
        sources = frozenset().union(*(dependencies[value] for value in node.inputs))
        dependencies[node.id] = sources
        if sources:
            # A parameter-dependent expression must be owned by this callee.
            if node.id in foreign_ids:
                dependencies.pop(node.id)
                continue
            movable.append(node)
    if not movable:
        return module
    movable_ids = {node.id for node in movable}
    boundary = set(function.outputs)
    for node in bodies[name]:
        if node.id not in movable_ids:
            boundary.update(node.inputs)
    roots = tuple(node for node in movable if node.id in boundary)
    parameters = {
        root.id: Node(
            f"{root.id}.invariant_parameter", "builtin.var", (), root.type,
            attrs={"name": f"invariant_{root.id}"},
            metadata={"function_parameter": name, "hoisted_call_invariant": root.id},
        ) for root in roots
    }
    redirects = {root: parameter.id for root, parameter in parameters.items()}
    remaining_parameters = tuple(value for value in function.parameters if value in boundary)
    refined = replace(
        function,
        parameters=(*remaining_parameters, *(parameters[root.id].id for root in roots)),
        outputs=tuple(redirects.get(value, value) for value in function.outputs),
    )
    call_owners = {call.id: owner for owner, calls in calls_by_caller.items() for call in calls}
    cache: dict[tuple[str, str, tuple[str, ...]], str] = {}
    result: list[Node] = []
    for node in module.nodes:
        if node.id in movable_ids:
            if node.id in parameters:
                result.append(parameters[node.id])
            continue
        owner = call_owners.get(node.id)
        if owner is None:
            result.append(replace(node, inputs=tuple(redirects.get(value, value) for value in node.inputs)))
            continue
        actuals = dict(zip(function.parameters, node.inputs, strict=True))
        for original in movable:
            inputs = tuple(actuals.get(value, value) for value in original.inputs)
            key = (owner, original.id, inputs)
            if key not in cache:
                clone = replace(
                    original, id=f"{node.id}.{original.id}.hoisted", inputs=inputs,
                    metadata={**original.metadata, "hoisted_from_function": name},
                )
                result.append(clone)
                cache[key] = clone.id
            actuals[original.id] = cache[key]
        result.append(replace(node, inputs=(
            *(actuals[value] for value in remaining_parameters),
            *(actuals[root.id] for root in roots),
        )))
    updated = replace(module, nodes=tuple(result), functions=tuple(
        refined if f.name == name else f for f in module.functions
    ))
    live = {node.id for f in updated.functions for node in function_nodes(updated, f)}
    return replace(updated, nodes=tuple(node for node in updated.nodes if node.id in live))


__all__ = ["hoist_call_invariant_expressions"]
