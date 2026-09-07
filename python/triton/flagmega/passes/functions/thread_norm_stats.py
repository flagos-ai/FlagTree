# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Thread normalization statistics through reusable graph functions.

This is the flat-SSA counterpart of nncase's
``ThreadNormStatsAcrossFunctionBoundariesPass``.  It changes only internal
function ABIs: the public entry ABI stays stable, while repeated calls can pass
the producer's additive statistics to the next invocation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import Function, IRModule, Node, PURE, TupleType, verify_module
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.nn._norm import normalize_axis
from triton.flagmega.ir.ops.nn.norm_stats import NormStats
from triton.flagmega.passes.functions.graph import function_nodes


@dataclass(frozen=True)
class _ThreadingPlan:
    function: Function
    input_parameter_index: int
    output_field_index: int
    axis: int
    use_mean: bool
    stats_type: object
    input_stats_ids: frozenset[str]
    stats_parameter_id: str
    stats_output_id: str

    @property
    def stats_output_index(self) -> int:
        return len(self.function.outputs)


def thread_norm_stats_across_function_boundaries(module: IRModule) -> IRModule:
    """Expose reusable-function NormStats as explicit input/output dataflow."""

    verify_module(module)
    users = _user_map(module)
    call_sites = _call_sites(module)
    occupied = {node.id for node in module.nodes}
    plans: dict[str, _ThreadingPlan] = {}
    for function in module.functions:
        if function.name == module.entry:
            continue
        calls = call_sites.get(function.name, ())
        if not calls:
            continue
        candidate = _create_plan(module, function, calls, users, occupied)
        if candidate is not None:
            plans[function.name] = candidate
            occupied.update((candidate.stats_parameter_id, candidate.stats_output_id))
    if not plans:
        return module

    node_map = module.node_map
    plans_by_call = {
        call.id: plans[str(call.attrs["callee"])]
        for calls in call_sites.values()
        for call in calls
        if str(call.attrs["callee"]) in plans
    }
    changed_roots = {
        node_id
        for plan in plans.values()
        for node_id in plan.input_stats_ids
    }
    # A source-level NormStats on a threaded state is also replaced by the
    # callee's appended stats projection.
    changed_roots.update(
        node.id
        for node in module.nodes
        if _threaded_stats_source(node, node_map, plans_by_call) is not None
    )

    new_parameters = {
        plan.stats_parameter_id: Node(
            plan.stats_parameter_id,
            "builtin.var",
            (),
            plan.stats_type,
            PURE,
            {"name": plan.stats_parameter_id},
            {"introduced_by": "ThreadNormStatsAcrossFunctionBoundaries"},
        )
        for plan in plans.values()
    }
    output_owner = {
        plan.function.outputs[plan.output_field_index]: plan
        for plan in plans.values()
    }

    nodes: list[Node] = list(new_parameters.values())
    created_ids = set(occupied)
    for node in module.nodes:
        if node.id in changed_roots:
            source = _threaded_stats_source(node, node_map, plans_by_call)
            if source is not None:
                call, plan = source
                nodes.append(Node(
                    node.id,
                    "builtin.get_item",
                    (call.id,),
                    plan.stats_type,
                    PURE,
                    {"index": plan.stats_output_index},
                    _threading_metadata(node),
                ))
            else:
                plan = next(value for value in plans.values() if node.id in value.input_stats_ids)
                nodes.append(Node(
                    node.id,
                    "nn.bind_norm_stats",
                    (node.inputs[0], plan.stats_parameter_id),
                    node.type,
                    node.effect,
                    node.attrs,
                    _threading_metadata(node),
                ))
        elif node.id in plans_by_call:
            plan = plans_by_call[node.id]
            input_id = node.inputs[plan.input_parameter_index]
            source = _threaded_value_source(input_id, node_map, plans_by_call)
            helper_id = _fresh_id(f"{node.id}.norm_stats", created_ids)
            created_ids.add(helper_id)
            if source is None:
                input_node = node_map[input_id]
                helper_type = NormStats.infer_type(
                    (input_node,), {"axis": plan.axis, "use_mean": plan.use_mean})
                helper = Node(
                    helper_id,
                    "nn.norm_stats",
                    (input_id,),
                    helper_type,
                    PURE,
                    {"axis": plan.axis, "use_mean": plan.use_mean},
                    {"introduced_by": "ThreadNormStatsAcrossFunctionBoundaries", "role": "seed"},
                )
            else:
                producer, producer_plan = source
                helper = Node(
                    helper_id,
                    "builtin.get_item",
                    (producer.id,),
                    producer_plan.stats_type,
                    PURE,
                    {"index": producer_plan.stats_output_index},
                    {"introduced_by": "ThreadNormStatsAcrossFunctionBoundaries", "role": "threaded"},
                )
            if helper.type != plan.stats_type:
                raise IRVerificationError(
                    f"Threaded statistics for call {node.id!r} have type {helper.type!r}, "
                    f"expected {plan.stats_type!r}.",
                    stage=module.stage,
                    node_id=node.id,
                )
            nodes.extend((helper, replace(
                node,
                inputs=(*node.inputs, helper.id),
                type=_append_tuple_field(node.type, plan.stats_type),
                metadata=_threading_metadata(node),
            )))
        else:
            nodes.append(node)

        plan = output_owner.get(node.id)
        if plan is not None:
            output_node = node_map[node.id]
            stats_type = NormStats.infer_type(
                (output_node,), {"axis": plan.axis, "use_mean": plan.use_mean})
            if stats_type != plan.stats_type:
                raise IRVerificationError(
                    f"Threaded output statistics of @{plan.function.name} have type {stats_type!r}, "
                    f"expected {plan.stats_type!r}.",
                    stage=module.stage,
                    node_id=node.id,
                )
            nodes.append(Node(
                plan.stats_output_id,
                "nn.norm_stats",
                (node.id,),
                stats_type,
                PURE,
                {"axis": plan.axis, "use_mean": plan.use_mean},
                {"introduced_by": "ThreadNormStatsAcrossFunctionBoundaries", "role": "output"},
            ))

    functions = tuple(
        replace(
            function,
            parameters=(*function.parameters, plans[function.name].stats_parameter_id),
            outputs=(*function.outputs, plans[function.name].stats_output_id),
        )
        if function.name in plans else function
        for function in module.functions
    )
    points = tuple(
        point for point in module.selection_points
        if point.owner not in changed_roots
    )
    point_ids = {point.id for point in points}
    result = replace(
        module,
        nodes=tuple(nodes),
        functions=functions,
        selection_points=points,
        selections=tuple(value for value in module.selections if value.point_id in point_ids),
    )
    return verify_module(result)


def _create_plan(
    module: IRModule,
    function: Function,
    calls: tuple[Node, ...],
    users: dict[str, tuple[Node, ...]],
    occupied: set[str],
) -> _ThreadingPlan | None:
    parameter_indices = {node_id: index for index, node_id in enumerate(function.parameters)}
    groups: dict[tuple[int, int, bool], list[Node]] = {}
    for node in function_nodes(module, function):
        if node.op != "nn.norm_stats" or node.inputs[0] not in parameter_indices:
            continue
        input_type = tensor_of(module.node_map[node.inputs[0]].type)
        axis = normalize_axis(int(node.attrs["axis"]), input_type.rank)
        key = (parameter_indices[node.inputs[0]], axis, bool(node.attrs["use_mean"]))
        groups.setdefault(key, []).append(node)
    if len(groups) != 1:
        return None
    (input_index, axis, use_mean), stats_nodes = next(iter(groups.items()))
    stats_type = stats_nodes[0].type
    if any(node.type != stats_type for node in stats_nodes):
        return None
    output_index = _state_output_index(
        module, function, calls, users,
        input_index=input_index,
        axis=axis,
        use_mean=use_mean,
    )
    if output_index is None:
        return None
    output_node = module.node_map[function.outputs[output_index]]
    if NormStats.infer_type((output_node,), {"axis": axis, "use_mean": use_mean}) != stats_type:
        return None
    parameter_id = _fresh_id(f"{function.parameters[input_index]}.norm_stats", occupied)
    output_id = _fresh_id(f"{function.name}.output_norm_stats", occupied | {parameter_id})
    return _ThreadingPlan(
        function,
        input_index,
        output_index,
        axis,
        use_mean,
        stats_type,
        frozenset(node.id for node in stats_nodes),
        parameter_id,
        output_id,
    )


def _state_output_index(
    module: IRModule,
    function: Function,
    calls: tuple[Node, ...],
    users: dict[str, tuple[Node, ...]],
    *,
    input_index: int,
    axis: int,
    use_mean: bool,
) -> int | None:
    output_count = len(function.outputs)
    if output_count == 1:
        return 0
    evidence: set[int] = set()
    for call in calls:
        call_users = users.get(call.id, ())
        if not call_users or any(
            value.op != "builtin.get_item"
            or value.inputs != (call.id,)
            or not 0 <= int(value.attrs["index"]) < output_count
            for value in call_users
        ):
            return None
        for projection in call_users:
            index = int(projection.attrs["index"])
            for consumer in users.get(projection.id, ()):
                if (
                    consumer.op == "builtin.call"
                    and str(consumer.attrs["callee"]) == function.name
                    and consumer.inputs[input_index] == projection.id
                ):
                    evidence.add(index)
                elif consumer.op == "nn.norm_stats" and consumer.inputs == (projection.id,):
                    input_type = tensor_of(projection.type)
                    candidate_axis = normalize_axis(int(consumer.attrs["axis"]), input_type.rank)
                    if candidate_axis == axis and bool(consumer.attrs["use_mean"]) == use_mean:
                        evidence.add(index)
    if len(evidence) == 1:
        return next(iter(evidence))
    input_type = module.node_map[function.parameters[input_index]].type
    type_matches = tuple(
        index for index, node_id in enumerate(function.outputs)
        if module.node_map[node_id].type == input_type
    )
    return type_matches[0] if not evidence and len(type_matches) == 1 else None


def _call_sites(module: IRModule) -> dict[str, tuple[Node, ...]]:
    result: dict[str, list[Node]] = {}
    live: set[str] = set()
    for function in module.functions:
        live.update(node.id for node in function_nodes(module, function))
    for node in module.nodes:
        if node.id in live and node.op == "builtin.call":
            result.setdefault(str(node.attrs["callee"]), []).append(node)
    return {key: tuple(value) for key, value in result.items()}


def _user_map(module: IRModule) -> dict[str, tuple[Node, ...]]:
    users: dict[str, list[Node]] = {node.id: [] for node in module.nodes}
    for node in module.nodes:
        for input_id in dict.fromkeys(node.inputs):
            users[input_id].append(node)
    return {key: tuple(value) for key, value in users.items()}


def _threaded_value_source(
    value_id: str,
    node_map: dict[str, Node],
    plans_by_call: dict[str, _ThreadingPlan],
) -> tuple[Node, _ThreadingPlan] | None:
    projection = node_map[value_id]
    if projection.op != "builtin.get_item":
        return None
    producer = node_map[projection.inputs[0]]
    plan = plans_by_call.get(producer.id)
    if plan is None or int(projection.attrs["index"]) != plan.output_field_index:
        return None
    return producer, plan


def _threaded_stats_source(
    node: Node,
    node_map: dict[str, Node],
    plans_by_call: dict[str, _ThreadingPlan],
) -> tuple[Node, _ThreadingPlan] | None:
    if node.op != "nn.norm_stats":
        return None
    source = _threaded_value_source(node.inputs[0], node_map, plans_by_call)
    if source is None:
        return None
    _, plan = source
    input_type = tensor_of(node_map[node.inputs[0]].type)
    axis = normalize_axis(int(node.attrs["axis"]), input_type.rank)
    return source if axis == plan.axis and bool(node.attrs["use_mean"]) == plan.use_mean else None


def _append_tuple_field(value_type, field_type):
    if isinstance(value_type, TupleType):
        return TupleType((*value_type.fields, field_type))
    return TupleType((value_type, field_type))


def _fresh_id(base: str, occupied: set[str]) -> str:
    if base not in occupied:
        return base
    index = 1
    while f"{base}_{index}" in occupied:
        index += 1
    return f"{base}_{index}"


def _threading_metadata(node: Node) -> dict[str, object]:
    return {
        **dict(node.metadata),
        "introduced_by": "ThreadNormStatsAcrossFunctionBoundaries",
    }


__all__ = ["thread_norm_stats_across_function_boundaries"]
