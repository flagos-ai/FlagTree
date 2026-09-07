# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Caller-aware legality of optional destructive aliases in reusable functions.

This analysis concerns pure tensor values, not explicit Ref state effects.
It preserves one ABI per function by requiring input donation to be legal at
every call site. Structural view/passthrough aliases are tracked independently
of SAT placement and optional arithmetic in-place decisions.
"""

from triton.flagmega.ir import RefType, TupleType, logical_type
from triton.flagmega.passes.functions.graph import callee_first_functions, function_nodes


def non_consumable_parameters(module) -> dict[str, frozenset[str]]:
    """Return formals whose original contents callers may still observe."""

    summaries = {}
    uses = {}
    protected = {function.name: set() for function in module.functions}
    for function in callee_first_functions(module):
        nodes = function_nodes(module, function)
        origins = {}
        calls = []
        readonly = set()
        last_use = {}
        for index, node in enumerate(nodes):
            if node.op in {"tir.buffer_view", "distributed.sharded_view"}:
                origin = origins[node.inputs[0]]
            elif node.op == "builtin.tuple":
                origin = tuple(origins[value] for value in node.inputs)
            elif node.op == "builtin.get_item":
                origin = origins[node.inputs[0]][int(node.attrs["index"])]
            elif node.op in {"builtin.call", "tir.call"} and str(node.attrs.get("callee", "")) in summaries:
                callee = module.function_map[str(node.attrs["callee"])]
                parameters, returned = summaries[callee.name]
                if len(returned) == 1 and logical_type(node.type) == logical_type(module.node_map[callee.outputs[0]].type):
                    returned = returned[0]
                substitutions = {}
                for formal, actual in zip(callee.parameters, node.inputs, strict=True):
                    _bind_origins(parameters[formal], origins[actual], substitutions)
                origin = _substitute(returned, substitutions, node.id)
                calls.append((index, callee, node.inputs))
            else:
                origin = _new_origins(node.id, node.type)
            origins[node.id] = origin
            if node.op in {"builtin.weight", "builtin.const_asset", "builtin.splat_const", "tir.buffer"}:
                readonly.update(_roots(origin))
            # A tuple projection reads only its chosen field. Reading the
            # container here would conflate independent tensor lifetimes.
            read_origins = (origin,) if node.op == "builtin.get_item" else tuple(origins[value] for value in node.inputs)
            for read_origin in read_origins:
                for root in _roots(read_origin):
                    last_use[root] = index
        for value in function.outputs:
            for root in _roots(origins[value]):
                last_use[root] = len(nodes)
        parameters = {value: origins[value] for value in function.parameters}
        returned = tuple(origins[value] for value in function.outputs)
        summaries[function.name] = (parameters, returned)
        uses[function.name] = (origins, calls, readonly, last_use, parameters)
        if function.name == module.entry:
            protected[function.name].update(
                value for value in function.parameters
                if not isinstance(logical_type(module.node_map[value].type), RefType)
            )

    # A wrapper parameter can be borrowed by its caller, and must then remain
    # borrowed when forwarded to another reusable function. Iterate the finite
    # parameter set to a fixed point; no call cloning or retry allocation.
    changed = True
    while changed:
        changed = False
        for name, (origins, calls, readonly, last_use, parameters) in uses.items():
            unavailable = set(readonly)
            for parameter in protected[name]:
                unavailable.update(_roots(parameters[parameter]))
            for index, callee, arguments in calls:
                for formal, actual in zip(callee.parameters, arguments, strict=True):
                    if isinstance(logical_type(module.node_map[formal].type), RefType):
                        continue
                    roots = _roots(origins[actual])
                    if (roots & unavailable or any(last_use.get(root, index) > index for root in roots)) and formal not in protected[callee.name]:
                        protected[callee.name].add(formal)
                        changed = True
    return {name: frozenset(values) for name, values in protected.items()}


def _new_origins(prefix, value_type):
    value_type = logical_type(value_type)
    if isinstance(value_type, TupleType):
        return tuple(_new_origins(f"{prefix}.{index}", field) for index, field in enumerate(value_type.fields))
    return frozenset((prefix,))


def _roots(origin):
    if isinstance(origin, frozenset):
        return origin
    return frozenset().union(*(_roots(value) for value in origin))


def _bind_origins(formal, actual, substitutions):
    if isinstance(formal, frozenset):
        for root in formal:
            substitutions[root] = _roots(actual)
    else:
        for lhs, rhs in zip(formal, actual, strict=True):
            _bind_origins(lhs, rhs, substitutions)


def _substitute(origin, substitutions, prefix):
    if isinstance(origin, frozenset):
        return frozenset().union(*(
            substitutions.get(root, frozenset((f"{prefix}:{root}",))) for root in origin
        ))
    return tuple(_substitute(value, substitutions, prefix) for value in origin)


__all__ = ["non_consumable_parameters"]
