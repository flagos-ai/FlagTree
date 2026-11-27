# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Solve canonical-coordinate constraints across views and graph calls."""

from __future__ import annotations

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.bufferization.leaves import buffer_leaves
from triton.flagmega.ir.distributed_type import is_fully_replicated, is_local_shard_subview
from triton.flagmega.ir.model import DistributedType, RefType, TensorType, TupleType, logical_type


def canonical_call_storage_values(module, required):
    """Preserve an actual view's coordinates in every callee and call site.

    Call argument types describe ownership, not whether their pointer addresses
    a compact component or a canonical parent allocation. Coordinate contracts
    therefore form constraints on physical leaves, including tuple projections,
    aliases, all call sites and structured loop carries. Solve these constraints
    before allocating either function; promotion after one call is order dependent.
    Partial/exclusive components never acquire a canonical coordinate contract.
    """
    nodes = module.node_map
    leaves = {}
    types = {}
    for node in module.nodes:
        kind = logical_type(node.type)
        if isinstance(kind, RefType) and kind.name == "parameter_bank":
            leaves[node.id] = ()
            continue
        if not isinstance(node.type, (TensorType, DistributedType, TupleType, RefType)):
            leaves[node.id] = ()
            continue
        values = tuple(buffer_leaves(node.id, node.type))
        leaves[node.id] = tuple(name for name, _, _ in values)
        types.update((name, value_type) for name, value_type, _ in values)

    def eligible(name):
        kind = types[name]
        return not isinstance(kind, DistributedType) or kind.partial is None and kind.exclusive is None

    parent = {name: name for name in types if eligible(name)}
    call_leaves = set()

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def connect(left, right):
        if len(left) != len(right):
            raise IRVerificationError(
                "Graph-call storage constraints require matching physical leaf arities.",
                stage=module.stage,
            )
        for before, after in zip(left, right, strict=True):
            if before in parent and after in parent:
                parent[find(after)] = find(before)

    seeds = set()
    for name, kind in types.items():
        if name in parent and isinstance(kind, DistributedType) and is_fully_replicated(kind):
            # An allocating broadcast producer feeding a graph call is also
            # canonical: function_argument storage retains the full tensor.
            seeds.add(name)
    for value in required:
        seeds.update(leaves.get(value, ()))
        if value in parent:
            seeds.add(value)
    for function in module.functions:
        for parameter in function.parameters:
            for name in leaves[parameter]:
                kind = types[name]
                if isinstance(kind, TensorType) or (
                    name in parent and is_fully_replicated(kind)
                ):
                    # Undistributed or broadcast parameters already own a
                    # complete logical allocation; a view retains its origin.
                    seeds.add(name)

    for node in module.nodes:
        result = leaves[node.id]
        if node.op == "builtin.tuple":
            connect(tuple(name for value in node.inputs for name in leaves[value]), result)
        elif node.op == "builtin.get_item":
            source = nodes[node.inputs[0]]
            if isinstance(source.type, TupleType):
                index = int(node.attrs["index"])
                counts = [len(buffer_leaves("field", kind)) for kind in source.type.fields]
                start = sum(counts[:index])
                connect(leaves[source.id][start:start + counts[index]], result)
        elif node.op in {"distributed.sharded_view", "tir.buffer_view", "tir.buffer_subspan"}:
            source = leaves[node.inputs[0]]
            connect(source, result)
            if node.op == "distributed.sharded_view" or node.attrs.get("alias_kind") == "sharded_view":
                for before, after in zip(source, result, strict=True):
                    incoming, outgoing = types[before], types[after]
                    if after not in parent or not isinstance(outgoing, DistributedType):
                        continue
                    if isinstance(incoming, TensorType) or (
                        before in parent and not is_local_shard_subview(incoming, outgoing)
                    ):
                        # Reassigning owners requires canonical backing even
                        # when the producer was initially a compact component.
                        seeds.add(after)
        elif node.op in {"builtin.call", "tir.call", "control.for"}:
            callee = module.function_map.get(str(node.attrs.get("callee", "")))
            if callee is None:
                continue
            formals, actuals = callee.parameters, node.inputs
            if node.op == "control.for":
                formals, actuals = formals[1:], actuals[3:]
                carried = int(node.attrs["iter_count"])
                for formal, output in zip(formals[:carried], callee.outputs, strict=True):
                    # The next trip passes each body result back as its
                    # corresponding iter argument; captures have no backedge.
                    connect(leaves[formal], leaves[output])
            for formal, actual in zip(formals, actuals, strict=True):
                connect(leaves[formal], leaves[actual])
                call_leaves.update(leaves[formal])
                call_leaves.update(leaves[actual])
            outputs = tuple(name for value in callee.outputs for name in leaves[value])
            connect(outputs, result)
            call_leaves.update(outputs)
            call_leaves.update(result)

    boundary_roots = {find(name) for name in call_leaves if name in parent}
    roots = {find(name) for name in seeds if name in parent} & boundary_roots
    return frozenset(required) | frozenset(name for name in parent if find(name) in roots)
