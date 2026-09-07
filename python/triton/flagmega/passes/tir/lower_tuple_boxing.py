# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Recursively realize tuple Boxing, matching nncase GenerateBoxingValue."""

from dataclasses import replace

from triton.flagmega.ir import IRModule, Node, TupleType


def lower_tuple_boxing(module: IRModule) -> IRModule:
    """Keep identity fields as aliases; select each changed tensor separately.

    Mixed TensorLoad/TensorStore/collective fields have different memory effects
    and must not borrow one sibling's kernel family or synchronization contract.
    This normalization precedes candidate selection and SAT bufferization, so
    ordinary GetItem/Tuple aliases retain each original MemSpan independently.
    """

    nodes: list[Node] = []
    node_map = dict(module.node_map)
    used = set(node_map)
    rewritten: set[str] = set()

    def fresh(base: str) -> str:
        result = base
        suffix = 0
        while result in used:
            suffix += 1
            result = f"{base}_{suffix}"
        used.add(result)
        return result

    def append(node: Node) -> Node:
        nodes.append(node)
        node_map[node.id] = node
        return node

    def field(source: Node, index: int, base: str) -> Node:
        if source.op == "builtin.tuple":
            return node_map[source.inputs[index]]
        return append(Node(
            fresh(base + ".input"), "builtin.get_item", (source.id,),
            source.type.fields[index], attrs={"index": index},
        ))

    def realize(source: Node, target, base: str, origin: Node) -> Node:
        if source.type == target:
            return source
        if isinstance(target, TupleType):
            outputs = tuple(
                realize(field(source, index, f"{base}.field_{index}"),
                        typ, f"{base}.field_{index}", origin)
                for index, typ in enumerate(target.fields)
            )
            return append(Node(fresh(base), "builtin.tuple", tuple(node.id for node in outputs), target))
        return append(Node(
            fresh(base), origin.op, (source.id,), target, origin.effect,
            attrs={"new_type": target},
            metadata={**dict(origin.metadata), "tuple_boxing_origin": origin.id},
        ))

    for node in module.nodes:
        if node.op not in {"distributed.boxing", "distributed.force_boxing"} or not isinstance(node.type, TupleType):
            append(node)
            continue
        source = node_map[node.inputs[0]]
        # Verified Boxing types already prove recursively matching structure
        # and logical tensor types. Retain the original result ID/return ABI.
        outputs = tuple(
            realize(field(source, index, f"{node.id}.field_{index}"),
                    typ, f"{node.id}.field_{index}", node)
            for index, typ in enumerate(node.type.fields)
        )
        append(replace(node, op="builtin.tuple", inputs=tuple(value.id for value in outputs), attrs={}))
        rewritten.add(node.id)
    if not rewritten:
        return module
    # Historical distribution decisions remain in the editable IR. A stale
    # semantic TIR proposal for a replaced tuple cannot be reused for a leaf.
    removed_points = {point.id for point in module.selection_points
                      if point.owner in rewritten and point.id.startswith("tir.")}
    return replace(
        module, nodes=tuple(nodes),
        selection_points=tuple(point for point in module.selection_points if point.id not in removed_points),
        selections=tuple(record for record in module.selections if record.point_id not in removed_points),
    )


__all__ = ["lower_tuple_boxing"]
