# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Sink materializing NormStats boxings into reusable functions.

This is the flat-SSA counterpart of nncase's
``SinkNormStatsBoxingAcrossFunctionBoundariesPass``.  AutoDistribution has
already chosen exact distributed types.  The pass relocates only a proven
P(Sum)-to-B statistics boxing, preserving graph semantics while exposing the
collective and ``NormApply`` to the same later TIR fusion boundary.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Mapping

from triton.flagmega.ir import (
    DistributedType,
    Function,
    IRModule,
    Node,
    ReduceOp,
    SBPBroadCast,
    SBPPartial,
    verify_module,
)
from triton.flagmega.ir.ops.nn.norm_stats import NormStats
from triton.flagmega.passes.functions.graph import function_nodes


@dataclass(frozen=True)
class _Consumer:
    binding_id: str | None
    source_parameter_index: int | None
    local_input_id: str
    axis: int
    use_mean: bool


@dataclass(frozen=True)
class _ParameterPlan:
    partial_type: DistributedType
    materialized_type: DistributedType


@dataclass(frozen=True)
class _ArgumentPlan:
    kind: str
    value_id: str
    partial_type: DistributedType
    consumer: _Consumer


@dataclass(frozen=True)
class _CallPlan:
    callee: str
    arguments: Mapping[int, _ArgumentPlan]


@dataclass(frozen=True)
class _Variant:
    source: Function
    name: str
    parameters: Mapping[int, _ParameterPlan]
    calls: tuple[str, ...]
    primary: bool


def sink_norm_stats_boxing_across_function_boundaries(module: IRModule) -> IRModule:
    """Move selected additive-statistics collectives across internal calls."""

    verify_module(module)
    owners = _node_owners(module)
    users = _user_map(module)
    calls_by_callee = _call_sites(module)
    variants: list[_Variant] = []
    call_plans: dict[str, _CallPlan] = {}

    for function in module.functions:
        if function.name == module.entry:
            continue
        calls = calls_by_callee.get(function.name, ())
        if not calls:
            continue
        candidates = _parameter_candidates(module, function, users)
        if not candidates:
            continue

        grouped: dict[
            tuple[tuple[int, DistributedType, DistributedType], ...],
            list[tuple[Node, dict[int, _ArgumentPlan]]],
        ] = {}
        for call in calls:
            parameters: dict[int, _ParameterPlan] = {}
            arguments: dict[int, _ArgumentPlan] = {}
            for index, (materialized_type, consumer) in candidates.items():
                argument = _partial_call_argument(
                    module,
                    call,
                    index,
                    materialized_type,
                    consumer,
                )
                if argument is None:
                    continue
                parameters[index] = _ParameterPlan(
                    argument.partial_type,
                    materialized_type,
                )
                arguments[index] = argument
            if not parameters:
                continue
            key = tuple(
                (index, plan.partial_type, plan.materialized_type)
                for index, plan in sorted(parameters.items())
            )
            grouped.setdefault(key, []).append((call, arguments))

        if not grouped:
            continue
        planned_count = sum(len(values) for values in grouped.values())
        keep_unspecialized_primary = planned_count != len(calls)
        suffix = 1
        for group_index, (key, planned) in enumerate(grouped.items()):
            primary = not keep_unspecialized_primary and group_index == 0
            name = function.name if primary else (
                f"{function.name}_norm_stats_layout_{suffix}"
            )
            if not primary:
                suffix += 1
            parameter_plans = {
                index: _ParameterPlan(partial, materialized)
                for index, partial, materialized in key
            }
            variant = _Variant(
                function,
                name,
                parameter_plans,
                tuple(call.id for call, _ in planned),
                primary,
            )
            variants.append(variant)
            for call, arguments in planned:
                call_plans[call.id] = _CallPlan(name, arguments)

    if not variants:
        return module

    primary_by_function = {
        variant.source.name: variant
        for variant in variants
        if variant.primary
    }
    occupied = {node.id for node in module.nodes}
    inserted_boxings: dict[tuple[str, int], Node] = {}
    for function_name, variant in primary_by_function.items():
        for index, plan in variant.parameters.items():
            parameter_id = variant.source.parameters[index]
            boxing_id = _fresh_id(f"{parameter_id}.materialized", occupied)
            occupied.add(boxing_id)
            inserted_boxings[(function_name, index)] = Node(
                boxing_id,
                "distributed.boxing",
                (parameter_id,),
                plan.materialized_type,
                attrs={"new_type": plan.materialized_type},
                metadata={
                    "introduced_by": "SinkNormStatsBoxingAcrossFunctionBoundaries",
                    "source_type": plan.partial_type,
                    "function": function_name,
                },
            )

    original_nodes: list[Node] = []
    removable_arguments: set[str] = set()
    for node in module.nodes:
        owner = _single_owner(owners, node.id)
        primary = primary_by_function.get(owner) if owner is not None else None
        rewritten = node
        if primary is not None:
            parameter_index = _parameter_index(primary.source, node.id)
            if parameter_index in primary.parameters:
                rewritten = replace(
                    rewritten,
                    type=primary.parameters[parameter_index].partial_type,
                    metadata={
                        **dict(rewritten.metadata),
                        "norm_stats_boundary": "partial",
                    },
                )
            substitutions = {
                primary.source.parameters[index]: inserted_boxings[(owner, index)].id
                for index in primary.parameters
            }
            rewritten = replace(
                rewritten,
                inputs=tuple(substitutions.get(value, value) for value in rewritten.inputs),
            )

        call_plan = call_plans.get(node.id)
        if call_plan is not None:
            rewritten, helpers, removed = _rewrite_call(
                node,
                rewritten,
                call_plan,
                module,
                occupied,
                input_map={
                    original: replacement
                    for original, replacement in (
                        zip(node.inputs, rewritten.inputs)
                    )
                },
            )
            original_nodes.extend(helpers)
            removable_arguments.update(removed)
        original_nodes.append(rewritten)

        if primary is not None:
            parameter_index = _parameter_index(primary.source, node.id)
            boxing = inserted_boxings.get((owner, parameter_index))
            if boxing is not None:
                original_nodes.append(boxing)

    extra_variants = tuple(variant for variant in variants if not variant.primary)
    clone_maps: dict[str, dict[str, str]] = {}
    cloned_nodes: list[Node] = []
    cloned_functions: list[Function] = []
    for variant in extra_variants:
        clone_map, nodes, function = _clone_variant(
            module,
            variant,
            owners,
            call_plans,
            occupied,
        )
        clone_maps[variant.name] = clone_map
        cloned_nodes.extend(nodes)
        cloned_functions.append(function)

    functions: list[Function] = []
    for function in module.functions:
        primary = primary_by_function.get(function.name)
        functions.append(
            replace(
                function,
                attrs={
                    **dict(function.attrs),
                    "norm_stats_boundary_specialization": tuple(
                        sorted(primary.parameters)
                    ),
                },
            )
            if primary is not None else function
        )
        functions.extend(
            variant_function
            for variant_function in cloned_functions
            if next(
                value.source.name
                for value in extra_variants
                if value.name == variant_function.name
            ) == function.name
        )

    points = list(module.selection_points)
    selections = list(module.selections)
    selection_map = {record.point_id: record for record in module.selections}
    for variant in extra_variants:
        clone_map = clone_maps[variant.name]
        for point in module.selection_points:
            if point.owner not in clone_map:
                continue
            point_id = _fresh_point_id(f"{point.id}.{variant.name}", {value.id for value in points})
            points.append(replace(point, id=point_id, owner=clone_map[point.owner]))
            record = selection_map.get(point.id)
            if record is not None:
                selections.append(replace(record, point_id=point_id))

    # Replaced call arguments are compiler-created roots.  Drop them only when
    # the transformed graph has no remaining use; unrelated dead/editable IR is
    # deliberately preserved.
    all_nodes = original_nodes + cloned_nodes
    remaining_users = _users_from_nodes(all_nodes)
    removed = {
        node_id for node_id in removable_arguments
        if not remaining_users.get(node_id)
        and all(node_id not in (*fn.parameters, *fn.outputs) for fn in functions)
    }
    if removed:
        all_nodes = [node for node in all_nodes if node.id not in removed]
        points = [point for point in points if point.owner not in removed]
        live_points = {point.id for point in points}
        selections = [record for record in selections if record.point_id in live_points]

    result = replace(
        module,
        nodes=tuple(all_nodes),
        functions=tuple(functions),
        selection_points=tuple(points),
        selections=tuple(selections),
    )
    return verify_module(result)


def _parameter_candidates(module, function, users):
    node_map = module.node_map
    body = {node.id for node in function_nodes(module, function)}
    result: dict[int, tuple[DistributedType, _Consumer]] = {}
    for index, parameter_id in enumerate(function.parameters):
        parameter = node_map[parameter_id]
        # Function results are semantic uses even though flat SSA represents
        # them on Function.outputs rather than as an explicit tuple node.
        if parameter_id in function.outputs:
            continue
        if not isinstance(parameter.type, DistributedType):
            continue
        materialized = parameter.type
        if materialized.partial is not None:
            continue
        body_users = tuple(node for node in users.get(parameter_id, ()) if node.id in body)
        if len(body_users) != 1 or body_users[0].inputs.count(parameter_id) != 1:
            continue
        first = body_users[0]
        if first.op == "nn.norm_apply" and first.inputs[1] == parameter_id:
            result[index] = (materialized, _Consumer(
                None,
                None,
                first.inputs[0],
                int(first.attrs["axis"]),
                bool(first.attrs["use_mean"]),
            ))
            continue
        if (
            first.op != "nn.bind_norm_stats"
            or first.inputs[1] != parameter_id
            or first.inputs[0] not in function.parameters
        ):
            continue
        binding_users = tuple(node for node in users.get(first.id, ()) if node.id in body)
        if (
            len(binding_users) != 1
            or binding_users[0].op != "nn.norm_apply"
            or binding_users[0].inputs[1] != first.id
            or binding_users[0].inputs.count(first.id) != 1
        ):
            continue
        apply = binding_users[0]
        if (
            int(first.attrs["axis"]) != int(apply.attrs["axis"])
            or bool(first.attrs["use_mean"]) != bool(apply.attrs["use_mean"])
        ):
            continue
        source_id = first.inputs[0]
        if not _is_sharded_view_of(module, apply.inputs[0], source_id):
            continue
        result[index] = (materialized, _Consumer(
            first.id,
            function.parameters.index(source_id),
            apply.inputs[0],
            int(first.attrs["axis"]),
            bool(first.attrs["use_mean"]),
        ))
    return result


def _partial_call_argument(
    module,
    call,
    parameter_index,
    materialized_type,
    consumer,
):
    argument = module.node_map[call.inputs[parameter_index]]
    if (
        argument.op == "distributed.boxing"
        and argument.type == materialized_type
        and argument.attrs["new_type"] == materialized_type
    ):
        source = module.node_map[argument.inputs[0]]
        if _is_supported_partial_type(source.type, materialized_type):
            return _ArgumentPlan("existing_partial", source.id, source.type, consumer)

    if (
        consumer.binding_id is None
        or consumer.source_parameter_index is None
        or argument.op != "nn.norm_stats"
        or argument.type != materialized_type
        or int(argument.attrs["axis"]) != consumer.axis
        or bool(argument.attrs["use_mean"]) != consumer.use_mean
        or argument.inputs[0] != call.inputs[consumer.source_parameter_index]
    ):
        return None
    local_type = _instantiated_local_type(module, consumer.local_input_id)
    local_node = Node("<local-input>", "builtin.var", (), local_type, attrs={"name": "local"})
    partial_type = NormStats.infer_type(
        (local_node,),
        {"axis": consumer.axis, "use_mean": consumer.use_mean},
    )
    if not _is_supported_partial_type(partial_type, materialized_type):
        return None
    return _ArgumentPlan("local_seed", argument.id, partial_type, consumer)


def _rewrite_call(original, rewritten, plan, module, occupied, *, input_map):
    inputs = list(rewritten.inputs)
    helpers: list[Node] = []
    removed: set[str] = set()
    for index, argument in plan.arguments.items():
        old_root = original.inputs[index]
        removed.add(old_root)
        if argument.kind == "existing_partial":
            inputs[index] = input_map.get(argument.value_id, argument.value_id)
            continue
        source_index = argument.consumer.source_parameter_index
        assert source_index is not None
        source_id = inputs[source_index]
        local_id = _clone_sharded_view_chain(
            module,
            argument.consumer.local_input_id,
            module.function_map[str(original.attrs["callee"])].parameters[source_index],
            source_id,
            f"{original.id}.norm_stats_seed",
            occupied,
            helpers,
            input_map=input_map,
        )
        stats_id = _fresh_id(f"{original.id}.partial_norm_stats", occupied)
        occupied.add(stats_id)
        helpers.append(Node(
            stats_id,
            "nn.norm_stats",
            (local_id,),
            argument.partial_type,
            attrs={"axis": argument.consumer.axis, "use_mean": argument.consumer.use_mean},
            metadata={
                "introduced_by": "SinkNormStatsBoxingAcrossFunctionBoundaries",
                "role": "partial_seed",
            },
        ))
        inputs[index] = stats_id
    return replace(
        rewritten,
        inputs=tuple(inputs),
        attrs={**dict(rewritten.attrs), "callee": plan.callee},
        metadata={
            **dict(rewritten.metadata),
            "norm_stats_boundary": "partial",
        },
    ), helpers, removed


def _clone_variant(module, variant, owners, call_plans, occupied):
    source_nodes = function_nodes(module, variant.source)
    private = {
        node.id for node in source_nodes
        if owners.get(node.id) == {variant.source.name}
        and (
            node.id in variant.source.parameters
            or node.inputs
            or node.op == "builtin.var"
        )
    }
    mapping: dict[str, str] = {}
    result: list[Node] = []
    materialized_by_parameter: dict[str, str] = {}
    suffix = variant.name
    for node in source_nodes:
        if node.id not in private:
            continue
        clone_id = _fresh_id(f"{node.id}.{suffix}", occupied)
        occupied.add(clone_id)
        mapping[node.id] = clone_id
        index = _parameter_index(variant.source, node.id)
        plan = variant.parameters.get(index)
        clone = replace(
            node,
            id=clone_id,
            inputs=tuple(mapping.get(value, value) for value in node.inputs),
            type=plan.partial_type if plan is not None else node.type,
            metadata={
                **dict(node.metadata),
                "cloned_for_function_variant": variant.name,
                **({"norm_stats_boundary": "partial"} if plan is not None else {}),
            },
        )
        if plan is not None:
            result.append(clone)
            boxing_id = _fresh_id(f"{clone_id}.materialized", occupied)
            occupied.add(boxing_id)
            boxing = Node(
                boxing_id,
                "distributed.boxing",
                (clone_id,),
                plan.materialized_type,
                attrs={"new_type": plan.materialized_type},
                metadata={
                    "introduced_by": "SinkNormStatsBoxingAcrossFunctionBoundaries",
                    "source_type": plan.partial_type,
                    "function": variant.name,
                },
            )
            result.append(boxing)
            materialized_by_parameter[node.id] = boxing_id
            continue

        substitutions = {
            parameter_id: materialized
            for parameter_id, materialized in materialized_by_parameter.items()
        }
        clone = replace(
            clone,
            inputs=tuple(substitutions.get(value, mapping.get(value, value)) for value in node.inputs),
        )
        call_plan = call_plans.get(node.id)
        if call_plan is not None:
            clone, helpers, _ = _rewrite_call(
                node,
                clone,
                call_plan,
                module,
                occupied,
                input_map={
                    value: substitutions.get(value, mapping.get(value, value))
                    for value in module.node_map
                },
            )
            result.extend(helpers)
        result.append(clone)

    parameters = tuple(mapping.get(value, value) for value in variant.source.parameters)
    outputs = tuple(mapping.get(value, value) for value in variant.source.outputs)
    function = Function(
        variant.name,
        parameters,
        outputs,
        {
            **dict(variant.source.attrs),
            "specialized_from": variant.source.name,
            "norm_stats_boundary_specialization": tuple(sorted(variant.parameters)),
        },
    )
    return mapping, result, function


def _clone_sharded_view_chain(
    module,
    value_id,
    source_parameter_id,
    replacement_id,
    prefix,
    occupied,
    output,
    *,
    input_map,
):
    if value_id == source_parameter_id:
        return replacement_id
    node = module.node_map[value_id]
    if node.op != "distributed.sharded_view" or len(node.inputs) != 1:
        raise ValueError(
            f"NormStats local input {value_id!r} is not a ShardedView chain."
        )
    parent = _clone_sharded_view_chain(
        module,
        node.inputs[0],
        source_parameter_id,
        replacement_id,
        prefix,
        occupied,
        output,
        input_map=input_map,
    )
    clone_id = _fresh_id(f"{prefix}.{node.id}", occupied)
    occupied.add(clone_id)
    output.append(replace(
        node,
        id=clone_id,
        inputs=(input_map.get(parent, parent),),
        metadata={
            **dict(node.metadata),
            "introduced_by": "SinkNormStatsBoxingAcrossFunctionBoundaries",
            "role": "partial_seed_view",
        },
    ))
    return clone_id


def _instantiated_local_type(module, value_id):
    node = module.node_map[value_id]
    if node.op == "distributed.sharded_view":
        return node.type
    return node.type


def _is_sharded_view_of(module, value_id, source_id):
    current = value_id
    while current != source_id:
        node = module.node_map[current]
        if node.op != "distributed.sharded_view" or len(node.inputs) != 1:
            return False
        current = node.inputs[0]
    return True


def _is_supported_partial_type(partial, materialized):
    return (
        isinstance(partial, DistributedType)
        and isinstance(materialized, DistributedType)
        and isinstance(partial.partial, SBPPartial)
        and partial.partial.reduce_op is ReduceOp.SUM
        and all(0 <= axis < partial.placement.rank for axis in partial.partial.axes)
        and all(isinstance(policy, SBPBroadCast) for policy in partial.axis_policies)
        and materialized.partial is None
        and all(isinstance(policy, SBPBroadCast) for policy in materialized.axis_policies)
        and partial.tensor == materialized.tensor
        and partial.placement == materialized.placement
    )


def _node_owners(module):
    result: dict[str, set[str]] = defaultdict(set)
    for function in module.functions:
        for node in function_nodes(module, function):
            result[node.id].add(function.name)
    return dict(result)


def _single_owner(owners, node_id):
    values = owners.get(node_id, set())
    return next(iter(values)) if len(values) == 1 else None


def _user_map(module):
    result: dict[str, list[Node]] = defaultdict(list)
    for node in module.nodes:
        for value in dict.fromkeys(node.inputs):
            result[value].append(node)
    return {key: tuple(value) for key, value in result.items()}


def _users_from_nodes(nodes):
    result: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        for value in dict.fromkeys(node.inputs):
            result[value].append(node.id)
    return result


def _call_sites(module):
    result: dict[str, list[Node]] = defaultdict(list)
    live = {
        node.id
        for function in module.functions
        for node in function_nodes(module, function)
    }
    for node in module.nodes:
        if node.id in live and node.op == "builtin.call":
            result[str(node.attrs["callee"])].append(node)
    return {key: tuple(value) for key, value in result.items()}


def _parameter_index(function, node_id):
    try:
        return function.parameters.index(node_id)
    except ValueError:
        return -1


def _fresh_id(base, occupied):
    if base not in occupied:
        return base
    index = 1
    while f"{base}_{index}" in occupied:
        index += 1
    return f"{base}_{index}"


def _fresh_point_id(base, occupied):
    return _fresh_id(base, occupied)


__all__ = ["sink_norm_stats_boxing_across_function_boundaries"]
