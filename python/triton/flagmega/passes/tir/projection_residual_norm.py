# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Legality analysis for projection/residual/RMS-stat epilogue fusion.

The analysis deliberately describes dataflow, not model layer names. A target
may offer a fused TIR candidate only when the projected tensor reaches exactly
one residual add (possibly through storage-view adapters), and that add feeds
exactly one RMSNorm source edge.  Other readers of the materialized residual
remain legal because the fused epilogue still publishes the add result.
"""

from __future__ import annotations

from dataclasses import dataclass

from triton.flagmega.ir import IRModule, Node, logical_type


_DIRECT_PROJECTION_OPS = frozenset({
    "math.matmul",
    "math.packed_dense_matmul",
    "math.block_scaled_matmul",
    "math.packed_block_scaled_matmul",
    "ntt.packed_matmul",
})
_TUPLE_PROJECTION_OPS = frozenset({
})
_PROJECTION_OPS = _DIRECT_PROJECTION_OPS | _TUPLE_PROJECTION_OPS
_TRANSPARENT_ADAPTERS = frozenset({
    "distributed.boxing",
    "distributed.sharded_view",
})


def _is_rms_consumer(node: Node, input_id: str) -> bool:
    return (
        node.op == "nn.rms_norm"
        and node.inputs
        and node.inputs[0] == input_id
    ) or (
        node.op == "nn.norm_apply"
        and len(node.inputs) == 4
        and node.inputs[0] == input_id
        and not bool(node.attrs["use_mean"])
    )


def _is_add(node: Node) -> bool:
    return node.op == "math.add" or (
        node.op == "math.vectorized_binary"
        and node.attrs.get("binary_op") == "add"
    )


@dataclass(frozen=True)
class ProjectionResidualNormMatch:
    producer: str
    projection_value: str
    adapters: tuple[str, ...]
    residual_add: str
    residual_input: str
    norm_consumer: str


def find_projection_residual_norm_matches(
    module: IRModule,
) -> dict[str, ProjectionResidualNormMatch]:
    """Return producer-keyed, exact single-projection-use fusion matches."""

    users: dict[str, list[Node]] = {node.id: [] for node in module.nodes}
    for node in module.nodes:
        for input_id in node.inputs:
            users.setdefault(input_id, []).append(node)

    matches: dict[str, ProjectionResidualNormMatch] = {}
    for producer in module.nodes:
        if producer.op not in _PROJECTION_OPS:
            continue
        projection = _projection_value(producer, users)
        if projection is None:
            continue
        add, adapters = _single_data_user(module, users, projection)
        if add is None or not _is_add(add) or add.inputs.count(adapters[-1] if adapters else projection.id) != 1:
            continue
        projected_input = adapters[-1] if adapters else projection.id
        residual_inputs = tuple(value for value in add.inputs if value != projected_input)
        if len(residual_inputs) != 1:
            continue
        norm_users = tuple(
            user for user in users.get(add.id, ())
            if _is_rms_consumer(user, add.id)
        )
        if len(norm_users) == 1:
            norm = norm_users[0]
        elif not norm_users:
            norm = _loop_carried_norm_consumer(module, users, producer, add)
            if norm is None:
                continue
        else:
            continue
        if (
            logical_type(projection.type) != logical_type(add.type)
            or logical_type(add.type) != logical_type(norm.type)
        ):
            continue
        matches[producer.id] = ProjectionResidualNormMatch(
            producer=producer.id,
            projection_value=projection.id,
            adapters=tuple(adapters),
            residual_add=add.id,
            residual_input=residual_inputs[0],
            norm_consumer=norm.id,
        )
    return matches


def _loop_carried_norm_consumer(
    module: IRModule,
    users: dict[str, list[Node]],
    producer: Node,
    residual_add: Node,
) -> Node | None:
    """Recognize a reusable function output normalized by its next call.

    This is the function-boundary form of the same projection/add/norm chain:
    the add is a callee output, intermediate calls feed it back to the callee's
    normalized first parameter, and the final call feeds an ordinary norm.
    No model or layer names participate in the proof.
    """

    functions = tuple(
        function
        for function in module.functions
        if bool(function.attrs.get("reusable"))
        and residual_add.id in function.outputs
        and function.parameters
    )
    if len(functions) != 1:
        return None
    function = functions[0]
    output_index = function.outputs.index(residual_add.id)
    parameter = function.parameters[0]
    entry_norms = _norms_reachable_through_adapters(parameter, users)
    if len(entry_norms) != 1:
        return None
    calls = tuple(
        node
        for node in module.nodes
        if node.op == "builtin.call"
        and str(node.attrs.get("callee")) == function.name
    )
    if not calls:
        return None
    projected: list[Node] = []
    for call in calls:
        fields = tuple(
            user
            for user in users.get(call.id, ())
            if user.op == "builtin.get_item"
            and int(user.attrs.get("index", -1)) == output_index
        )
        if len(fields) != 1:
            return None
        projected.append(fields[0])
    for index, value in enumerate(projected[:-1]):
        if not calls[index + 1].inputs or calls[index + 1].inputs[0] != value.id:
            return None
        if users.get(value.id, ()) != [calls[index + 1]]:
            return None
    final_norms = tuple(
        user
        for user in users.get(projected[-1].id, ())
        if _is_rms_consumer(user, projected[-1].id)
    )
    if len(final_norms) != 1:
        return None
    return entry_norms[0]


def _norms_reachable_through_adapters(
    value_id: str,
    users: dict[str, list[Node]],
) -> tuple[Node, ...]:
    pending = [value_id]
    visited: set[str] = set()
    norms: dict[str, Node] = {}
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        for user in users.get(current, ()):
            if user.op in _TRANSPARENT_ADAPTERS and user.inputs == (current,):
                pending.append(user.id)
            elif _is_rms_consumer(user, current):
                norms[user.id] = user
    return tuple(norms.values())


def _projection_value(
    producer: Node,
    users: dict[str, list[Node]],
) -> Node | None:
    if producer.op in _DIRECT_PROJECTION_OPS:
        return producer
    outputs = tuple(
        user for user in users.get(producer.id, ())
        if user.op == "builtin.get_item" and int(user.attrs.get("index", -1)) == 0
    )
    return outputs[0] if len(outputs) == 1 else None


def _single_data_user(
    module: IRModule,
    users: dict[str, list[Node]],
    value: Node,
) -> tuple[Node | None, list[str]]:
    current = value
    adapters: list[str] = []
    while True:
        current_users = users.get(current.id, ())
        if len(current_users) != 1:
            return None, adapters
        user = current_users[0]
        if user.op not in _TRANSPARENT_ADAPTERS:
            return user, adapters
        if len(user.inputs) != 1 or logical_type(user.type) != logical_type(current.type):
            return None, adapters
        adapters.append(user.id)
        current = module.node_map[user.id]


__all__ = [
    "ProjectionResidualNormMatch",
    "find_projection_residual_norm_matches",
]
