# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Outline constant-parameter expressions into their actual caller contexts.

Unlike call invariance, constness does not require equal weights across calls.
Unlike packing, it does not depend on a marker on the expression root. The
complete call graph proves constant arguments; one refined callee ABI accepts
the transformed values, which ordinary constant freezing materializes offline.
"""

from __future__ import annotations

from dataclasses import replace

from triton.flagmega.errors import IRVerificationError, StageError
from triton.flagmega.ir import DistributedType, IRModule, Node, PointerType, TensorType, TupleType, verify_module
from triton.flagmega.ir.ops.core import get_definition
from triton.flagmega.passes.constants import ConstnessAnalysis, require_constants_open
from .graph import callee_first_functions, function_nodes


def _storage_value(value_type) -> bool:
    if isinstance(value_type, DistributedType):
        return value_type.partial is None and _storage_value(value_type.tensor)
    if isinstance(value_type, TensorType):
        return not isinstance(value_type.dtype, PointerType) and all(d.is_fixed for d in value_type.shape)
    return (isinstance(value_type, TupleType) and bool(value_type.fields)
            and all(_storage_value(field) for field in value_type.fields))


def _evaluable(node: Node) -> bool:
    definition = get_definition(node.op)
    return (node.effect.is_pure and definition.const_evaluable and definition.deterministic
            and _storage_value(node.type))


def _constant_values(module: IRModule) -> set[str]:
    """Monotone interprocedural proof, including constants passed through wrappers."""
    known = set(ConstnessAnalysis.analyze(module).constants)
    node_map, functions = module.node_map, module.function_map
    calls = {f.name: [] for f in module.functions if f.name != module.entry}
    for node in module.nodes:
        if node.op == "builtin.call" and node.attrs["callee"] in calls:
            calls[node.attrs["callee"]].append(node)
    while True:
        previous = len(known)
        for name, invocations in calls.items():
            if not invocations:
                continue
            for index, value in enumerate(functions[name].parameters):
                if _storage_value(node_map[value].type) and all(c.inputs[index] in known for c in invocations):
                    known.add(value)
        for node in module.nodes:
            if _evaluable(node) and node.inputs and all(value in known for value in node.inputs):
                known.add(node.id)
        if len(known) == previous:
            return known


def lift_constant_parameter_expressions(module: IRModule) -> IRModule:
    require_constants_open(module, "LiftConstantParameterExpressions")
    if module.prim_functions or any(n.op == "tir.call" for n in module.nodes):
        raise StageError("Constant parameter lifting must precede TIR lowering.", stage=module.stage)
    verify_module(module)
    # Move an inner expression into its wrapper first, then into the entry.
    for function in callee_first_functions(module):
        if function.name != module.entry:
            module = _lift_function(module, function.name)
    return verify_module(module)


def _lift_function(module: IRModule, name: str) -> IRModule:
    function = module.function_map[name]
    known = _constant_values(module)
    constant_parameters = set(function.parameters) & known
    if not constant_parameters:
        return module
    bodies = {f.name: function_nodes(module, f) for f in module.functions}
    foreign = {n.id for owner, body in bodies.items() if owner != name for n in body}
    global_constants = ConstnessAnalysis.analyze(module).constants
    dependencies = {value: frozenset() for value in global_constants}
    dependencies.update({value: frozenset({value}) for value in constant_parameters})
    movable = []
    for node in bodies[name]:
        if node.id in function.parameters or not _evaluable(node) or any(v not in dependencies for v in node.inputs):
            continue
        sources = frozenset().union(*(dependencies[v] for v in node.inputs))
        if sources and node.id in foreign:
            continue
        dependencies[node.id] = sources
        if sources:
            movable.append(node)
    if not movable:
        return module
    moved = {n.id for n in movable}
    boundary = set(function.outputs)
    for node in bodies[name]:
        if node.id not in moved:
            boundary.update(node.inputs)
    roots = tuple(n for n in movable if n.id in boundary)
    parameters = {
        n.id:
        Node(f"{n.id}.constant_parameter", "builtin.var", (), n.type, attrs={"name": f"constant_{n.id}"},
             metadata={"function_parameter": name, "lifted_constant_expression": n.id})
        for n in roots
    }
    retained = tuple(value for value in function.parameters if value in boundary)
    redirects = {value: parameter.id for value, parameter in parameters.items()}
    refined = replace(function, parameters=(*retained, *(parameters[n.id].id for n in roots)),
                      outputs=tuple(redirects.get(value, value) for value in function.outputs))
    known_ids = set(module.node_map)
    for parameter in parameters.values():
        _claim_id(parameter.id, known_ids, module.stage)
    rewritten_calls = set()
    result = []
    for node in module.nodes:
        if node.id in moved:
            if node.id in parameters:
                result.append(parameters[node.id])
            continue
        if node.op != "builtin.call" or node.attrs["callee"] != name:
            result.append(replace(node, inputs=tuple(redirects.get(v, v) for v in node.inputs)))
            continue
        actuals = dict(zip(function.parameters, node.inputs, strict=True))
        for original in movable:
            identity = f"{node.id}.{original.id}.constant"
            _claim_id(identity, known_ids, module.stage)
            clone = replace(original, id=identity, inputs=tuple(actuals.get(v, v) for v in original.inputs),
                            metadata={**original.metadata, "lifted_constant_from_function": name})
            result.append(clone)
            actuals[original.id] = identity
        result.append(replace(node, inputs=(*(actuals[v] for v in retained), *(actuals[n.id] for n in roots))))
        rewritten_calls.add(node.id)
    updated = replace(module, nodes=_order_constant_dependencies(result, global_constants, module.stage),
                      functions=tuple(refined if f.name == name else f for f in module.functions))
    live = {n.id for f in updated.functions for n in function_nodes(updated, f)}
    # Choices on moved/changed expressions have been realized in their typed
    # clones; old owners no longer describe selectable runtime computations.
    invalid = moved | rewritten_calls | (set(function.parameters) - set(retained))
    points = tuple(p for p in module.selection_points
                   if p.owner not in invalid and (p.owner is None or p.owner in live))
    point_ids = {p.id for p in points}
    return replace(updated, nodes=tuple(n for n in updated.nodes if n.id in live), selection_points=points,
                   selections=tuple(s for s in module.selections if s.point_id in point_ids))


def _claim_id(identity: str, known_ids: set[str], stage: str) -> None:
    if identity in known_ids:
        raise IRVerificationError(f"Constant parameter lifting node name collides: {identity!r}.", stage=stage)
    known_ids.add(identity)


def _order_constant_dependencies(nodes, constants, stage):
    """A callee may follow its caller; pull only proven constant dependencies.

    Nonconstant nodes retain their relative order, including effectful calls.
    This is not a general graph reorder that could hide a broken SSA rewrite.
    """
    node_map = {node.id: node for node in nodes}
    emitted, result = set(), []

    def emit(node):
        if node.id in emitted:
            return
        for value in node.inputs:
            if value not in emitted:
                if value not in constants or value not in node_map:
                    raise IRVerificationError(f"Lifted expression {node.id!r} precedes nonconstant input {value!r}.",
                                              stage=stage)
                emit(node_map[value])
        result.append(node)
        emitted.add(node.id)

    for node in nodes:
        emit(node)
    return tuple(result)


__all__ = ["lift_constant_parameter_expressions"]
