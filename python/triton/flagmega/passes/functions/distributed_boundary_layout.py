# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Plan distributed layouts across reusable function boundaries.

This is the post-AutoDistribution counterpart of nncase's
``FunctionBoundaryLayoutPropagationPass``.  It retains nncase's module-level
shape: identity, callee-internal, and caller-demand variants are selected with
CP-SAT, every call is rewritten to one shared ABI, and the rewrite iterates to
a fixed point.  Target policy decides only how a semantic reshard is realized
(``Boxing`` or a read-only ``ShardedView``).

Unlike nncase's pre-distribution input analysis, this pass can also promote an
already distributed parameter to another distributed layout.  That extension
is required because FlagMega runs AutoDistribution before this stage.  Raw
uses retain their original type through an explicit inverse reshard, while
other distributed adapters reshard directly from the promoted ABI.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from typing import Mapping

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import (
    DistributedType,
    Function,
    IRModule,
    IRType,
    Node,
    PURE,
    SBPPartial,
    TensorType,
    TupleType,
    verify_module,
)
from triton.flagmega.passes.auto_distributed.realization import (
    DistributedReshardRealization,
    DistributedReshardRealizationContext,
    DistributedReshardSourceKind,
    DistributedReshardUsageKind,
)
from triton.flagmega.passes.constants import ConstnessAnalysis
from triton.flagmega.passes.functions.graph import function_nodes


_DISTRIBUTED_ADAPTERS = frozenset({
    "distributed.boxing",
    "distributed.sharded_view",
})
_MAX_PLANNING_ITERATIONS = 8
_IDENTITY_VARIANT_PENALTY = 1000
_LOCAL_TRANSFORM_COST_SCALE = 10


@dataclass(frozen=True)
class _InputLayout:
    index: int
    parameter_id: str
    source_type: IRType
    target_type: DistributedType
    adapter_ids: frozenset[str]


@dataclass(frozen=True)
class _OutputLayout:
    index: int
    source_id: str
    logical_type: IRType
    abi_type: IRType
    internal_adapter_id: str | None = None
    source_transform: bool = False
    restore_at_caller: bool = True
    abi_id: str | None = None


@dataclass(frozen=True)
class _FunctionLayout:
    function: Function
    inputs: dict[int, _InputLayout]
    outputs: dict[int, _OutputLayout]
    candidate_id: str = "internal"
    is_identity: bool = False


def propagate_post_auto_distributed_function_boundary_layouts(
    module: IRModule,
    realization_policy,
) -> IRModule:
    """Select and propagate distributed ABIs to a fixed point."""

    current = verify_module(module)
    for iteration in range(_MAX_PLANNING_ITERATIONS):
        rewritten = _propagate_one_iteration(
            current,
            realization_policy,
            enable_caller_output_demand=iteration == 0,
        )
        if rewritten is current:
            return current
        current = rewritten

    constants = ConstnessAnalysis.analyze(current).constants
    remaining = _collect_candidate_layouts(
        current,
        realization_policy,
        constants,
        enable_caller_output_demand=False,
    )
    if remaining:
        details = ", ".join(
            f"{name}: {[candidate.candidate_id for candidate in candidates]}"
            for name, candidates in sorted(remaining.items())
        )
        raise IRVerificationError(
            "Post-AutoDistribution function-boundary planning did not "
            f"converge within {_MAX_PLANNING_ITERATIONS} iterations; "
            f"remaining variants: {details}.",
            stage=current.stage,
        )
    return current


def _propagate_one_iteration(
    module: IRModule,
    realization_policy,
    *,
    enable_caller_output_demand: bool,
) -> IRModule:
    module = verify_module(module)
    constants = ConstnessAnalysis.analyze(module).constants
    users = _user_map(module)
    calls = _calls_by_callee(module)
    candidates = _collect_candidate_layouts(
        module,
        realization_policy,
        constants,
        users=users,
        calls=calls,
        enable_caller_output_demand=enable_caller_output_demand,
    )
    layouts = _select_layouts(module, candidates, users, calls)
    if not layouts:
        return module

    occupied = {node.id for node in module.nodes}
    node_map = module.node_map

    # Allocate callee-local ABI producers only for selected caller-demand
    # variants.  Internal variants reuse the source of the stripped adapter.
    for function_name, layout in tuple(layouts.items()):
        outputs = {}
        for index, output in layout.outputs.items():
            if output.source_transform:
                abi_id = _fresh_id(
                    f"{output.source_id}.distributed_boundary_abi", occupied)
                occupied.add(abi_id)
                output = replace(output, abi_id=abi_id)
            else:
                output = replace(output, abi_id=output.source_id)
            outputs[index] = output
        layouts[function_name] = replace(layout, outputs=outputs)

    function_output_roots = {
        output for function in module.functions for output in function.outputs
    }

    # A selected output ABI needs a raw typed value at every call.  Single
    # outputs use a renamed call; tuple outputs use a renamed GetItem.  When a
    # tuple call itself has a logical consumer, preserve its old type with an
    # explicit per-field wrapper exactly like nncase's WrapOutputs.
    raw_projection_ids: dict[str, str] = {}
    projection_layouts: dict[str, _OutputLayout] = {}
    tuple_raw_call_ids: dict[str, str] = {}
    for node in module.nodes:
        if node.op == "builtin.call":
            layout = layouts.get(str(node.attrs["callee"]))
            if layout is not None and len(layout.function.outputs) == 1:
                output = layout.outputs.get(0)
                if output is not None:
                    raw_id = _fresh_id(
                        f"{node.id}.distributed_boundary_abi", occupied)
                    occupied.add(raw_id)
                    raw_projection_ids[node.id] = raw_id
                    projection_layouts[node.id] = output
            elif layout is not None and layout.outputs and (
                node.id in function_output_roots
                or any(
                    user.op != "builtin.get_item"
                    for user in users.get(node.id, ())
                )
            ):
                raw_id = _fresh_id(
                    f"{node.id}.distributed_boundary_abi", occupied)
                occupied.add(raw_id)
                tuple_raw_call_ids[node.id] = raw_id
            continue
        if node.op != "builtin.get_item" or len(node.inputs) != 1:
            continue
        producer = node_map[node.inputs[0]]
        if producer.op != "builtin.call":
            continue
        layout = layouts.get(str(producer.attrs["callee"]))
        if layout is None:
            continue
        output = layout.outputs.get(int(node.attrs["index"]))
        if output is None:
            continue
        raw_id = _fresh_id(
            f"{node.id}.distributed_boundary_abi", occupied)
        occupied.add(raw_id)
        raw_projection_ids[node.id] = raw_id
        projection_layouts[node.id] = output

    removed_substitutions: dict[str, str] = {}
    removed_owners: dict[str, str] = {}
    parameter_layouts: dict[str, _InputLayout] = {}
    for layout in layouts.values():
        for item in layout.inputs.values():
            parameter_layouts[item.parameter_id] = item
            for adapter_id in item.adapter_ids:
                removed_substitutions[adapter_id] = item.parameter_id
                removed_owners[adapter_id] = item.parameter_id
        for output in layout.outputs.values():
            if output.internal_adapter_id is not None:
                removed_substitutions[output.internal_adapter_id] = output.source_id
                removed_owners[output.internal_adapter_id] = output.source_id

    # A caller transform that asks for the selected ABI consumes the raw call
    # directly.  Other consumers retain a compatibility reshard.
    for projection_id, raw_id in raw_projection_ids.items():
        output = projection_layouts[projection_id]
        for consumer in users.get(projection_id, ()):
            if (
                consumer.op in _DISTRIBUTED_ADAPTERS
                and len(consumer.inputs) == 1
                and consumer.type == output.abi_type
            ):
                removed_substitutions[consumer.id] = raw_id
                removed_owners[consumer.id] = raw_id

    raw_parameter_restores: dict[str, Node] = {}
    raw_parameter_use_nodes: dict[str, frozenset[str]] = {}
    for function_name, layout in layouts.items():
        owned = {node.id for node in function_nodes(module, layout.function)}
        for item in layout.inputs.values():
            raw_users = frozenset(
                user.id
                for user in users.get(item.parameter_id, ())
                if user.id in owned and user.op not in _DISTRIBUTED_ADAPTERS
            )
            output_needs_logical_source = (
                item.parameter_id in layout.function.outputs
                or any(
                    output.source_id == item.parameter_id
                    for output in layout.outputs.values()
                )
            )
            if not raw_users and not output_needs_logical_source:
                continue
            restore_id = _fresh_id(
                f"{item.parameter_id}.distributed_boundary_restore", occupied)
            occupied.add(restore_id)
            raw_parameter_restores[item.parameter_id] = _make_adapter_node(
                restore_id,
                item.parameter_id,
                item.target_type,
                item.source_type,
                DistributedReshardSourceKind.FUNCTION_PARAMETER,
                DistributedReshardUsageKind.INTERNAL,
                realization_policy,
                metadata={
                    "introduced_by": (
                        "PropagatePostAutoDistributedFunctionBoundaryLayouts"
                    ),
                    "boundary_layout": "raw_parameter_restore",
                    "callee": function_name,
                    "parameter_index": item.index,
                },
            )
            raw_parameter_use_nodes[item.parameter_id] = raw_users

    source_helpers: dict[str, list[Node]] = {}
    for function_name, layout in layouts.items():
        for output in layout.outputs.values():
            if not output.source_transform:
                continue
            assert output.abi_id is not None
            source = node_map[output.source_id]
            helper_input = (
                raw_parameter_restores[output.source_id].id
                if output.source_id in raw_parameter_restores
                else output.source_id
            )
            source_helpers.setdefault(output.source_id, []).append(
                _make_adapter_node(
                    output.abi_id,
                    helper_input,
                    output.logical_type,
                    output.abi_type,
                    _source_kind(source, constants),
                    DistributedReshardUsageKind.INTERNAL,
                    realization_policy,
                    metadata={
                        "introduced_by": (
                            "PropagatePostAutoDistributedFunctionBoundaryLayouts"
                        ),
                        "boundary_layout": "distributed_output_abi",
                        "callee": function_name,
                        "output_index": output.index,
                    },
                )
            )

    nodes: list[Node] = []
    for original in module.nodes:
        if original.id in removed_substitutions:
            continue

        if original.id in parameter_layouts:
            item = parameter_layouts[original.id]
            nodes.append(replace(
                original,
                type=item.target_type,
                metadata={
                    **dict(original.metadata),
                    "boundary_layout": "distributed",
                    "introduced_by": (
                        "PropagatePostAutoDistributedFunctionBoundaryLayouts"
                    ),
                },
            ))
            if original.id in raw_parameter_restores:
                nodes.append(raw_parameter_restores[original.id])
            nodes.extend(source_helpers.get(original.id, ()))
            continue

        if original.op == "builtin.call" and str(original.attrs["callee"]) in layouts:
            layout = layouts[str(original.attrs["callee"])]
            arguments = list(original.inputs)
            helpers: list[Node] = []
            for index, item in layout.inputs.items():
                argument_id = arguments[index]
                projected_raw = raw_projection_ids.get(argument_id)
                if (
                    projected_raw is not None
                    and projection_layouts[argument_id].abi_type == item.target_type
                ):
                    arguments[index] = projected_raw
                    continue
                argument = node_map[argument_id]
                if argument.type == item.target_type:
                    arguments[index] = _resolve(
                        argument_id, removed_substitutions)
                    continue
                if (
                    argument.op in _DISTRIBUTED_ADAPTERS
                    and len(argument.inputs) == 1
                    and node_map[argument.inputs[0]].type == item.target_type
                ):
                    arguments[index] = _resolve(
                        argument.inputs[0], removed_substitutions)
                    continue
                resolved_id = _resolve(argument_id, removed_substitutions)
                resolved = node_map.get(resolved_id, argument)
                helper_id = _fresh_id(
                    f"{original.id}.arg{index}.distributed_boundary_reshard",
                    occupied,
                )
                occupied.add(helper_id)
                helpers.append(_make_adapter_node(
                    helper_id,
                    resolved_id,
                    resolved.type,
                    item.target_type,
                    _source_kind(resolved, constants),
                    DistributedReshardUsageKind.FUNCTION_BOUNDARY,
                    realization_policy,
                    metadata={
                        "introduced_by": (
                            "PropagatePostAutoDistributedFunctionBoundaryLayouts"
                        ),
                        "callee": layout.function.name,
                        "parameter_index": index,
                    },
                ))
                arguments[index] = helper_id
            nodes.extend(helpers)
            rewritten_call = replace(
                original,
                inputs=tuple(arguments),
                type=_result_type_after_layout(module, layout),
                metadata={
                    **dict(original.metadata),
                    "introduced_by": (
                        "PropagatePostAutoDistributedFunctionBoundaryLayouts"
                    ),
                },
            )
            if original.id in tuple_raw_call_ids:
                raw_call = replace(
                    rewritten_call,
                    id=tuple_raw_call_ids[original.id],
                    metadata={
                        **dict(rewritten_call.metadata),
                        "boundary_layout": "distributed_abi",
                    },
                )
                nodes.append(raw_call)
                wrapper_fields = []
                assert isinstance(original.type, TupleType)
                assert isinstance(raw_call.type, TupleType)
                for field_index, (raw_type, logical_type) in enumerate(zip(
                    raw_call.type.fields, original.type.fields
                )):
                    field_id = _fresh_id(
                        f"{original.id}.distributed_boundary_field{field_index}",
                        occupied,
                    )
                    occupied.add(field_id)
                    nodes.append(Node(
                        field_id,
                        "builtin.get_item",
                        (raw_call.id,),
                        raw_type,
                        PURE,
                        {"index": field_index},
                        {
                            "introduced_by": (
                                "PropagatePostAutoDistributedFunctionBoundaryLayouts"
                            ),
                            "boundary_layout": "distributed_abi",
                        },
                    ))
                    output_layout = layout.outputs.get(field_index)
                    if output_layout is None or raw_type == logical_type:
                        wrapper_fields.append(field_id)
                        continue
                    if not output_layout.restore_at_caller:
                        raise IRVerificationError(
                            f"Irreversible output {field_index} of call "
                            f"{original.id!r} has a direct tuple consumer.",
                            stage=module.stage,
                            node_id=original.id,
                        )
                    logical_id = _fresh_id(
                        f"{field_id}.logical", occupied)
                    occupied.add(logical_id)
                    nodes.append(_make_adapter_node(
                        logical_id,
                        field_id,
                        raw_type,
                        logical_type,
                        DistributedReshardSourceKind.INTERNAL,
                        DistributedReshardUsageKind.INTERNAL,
                        realization_policy,
                        metadata={
                            "introduced_by": (
                                "PropagatePostAutoDistributedFunctionBoundaryLayouts"
                            ),
                            "boundary_layout": "logical_view",
                        },
                    ))
                    wrapper_fields.append(logical_id)
                nodes.append(Node(
                    original.id,
                    "builtin.tuple",
                    tuple(wrapper_fields),
                    original.type,
                    PURE,
                    {},
                    {
                        **dict(original.metadata),
                        "introduced_by": (
                            "PropagatePostAutoDistributedFunctionBoundaryLayouts"
                        ),
                        "boundary_layout": "logical_tuple_view",
                    },
                ))
                removed_owners[original.id] = raw_call.id
                continue
            if original.id not in raw_projection_ids:
                nodes.append(rewritten_call)
                continue
            output = projection_layouts[original.id]
            raw = replace(
                rewritten_call,
                id=raw_projection_ids[original.id],
                type=output.abi_type,
                metadata={
                    **dict(rewritten_call.metadata),
                    "boundary_layout": "distributed_abi",
                },
            )
            nodes.append(raw)
            if _needs_logical_projection(
                original,
                output,
                users,
                removed_substitutions,
                layouts,
                function_output_roots,
            ):
                nodes.append(_make_compatibility_view(
                    original,
                    raw.id,
                    output,
                    realization_policy,
                ))
            else:
                removed_substitutions[original.id] = raw.id
                removed_owners[original.id] = raw.id
            continue

        if original.id in raw_projection_ids:
            output = projection_layouts[original.id]
            projection_inputs = tuple(
                tuple_raw_call_ids.get(value, _resolve(value, removed_substitutions))
                for value in original.inputs
            )
            raw = replace(
                original,
                id=raw_projection_ids[original.id],
                inputs=projection_inputs,
                type=output.abi_type,
                metadata={
                    **dict(original.metadata),
                    "introduced_by": (
                        "PropagatePostAutoDistributedFunctionBoundaryLayouts"
                    ),
                    "boundary_layout": "distributed_abi",
                },
            )
            nodes.append(raw)
            if _needs_logical_projection(
                original,
                output,
                users,
                removed_substitutions,
                layouts,
                function_output_roots,
            ):
                nodes.append(_make_compatibility_view(
                    original,
                    raw.id,
                    output,
                    realization_policy,
                ))
            else:
                removed_substitutions[original.id] = raw.id
                removed_owners[original.id] = raw.id
            continue

        # A non-selected distributed use of a promoted parameter reshards
        # directly from the new ABI.  It is not a raw logical use and must not
        # take a detour through the compatibility restore.
        if (
            original.op in _DISTRIBUTED_ADAPTERS
            and len(original.inputs) == 1
            and original.inputs[0] in parameter_layouts
        ):
            parameter_layout = parameter_layouts[original.inputs[0]]
            realization = _classify_types(
                parameter_layout.target_type,
                original.type,
                DistributedReshardSourceKind.FUNCTION_PARAMETER,
                DistributedReshardUsageKind.INTERNAL,
                realization_policy,
            )
            if realization is DistributedReshardRealization.UNSUPPORTED:
                raise IRVerificationError(
                    f"Cannot realize promoted distributed parameter "
                    f"{parameter_layout.target_type!r} -> {original.type!r}.",
                    stage=module.stage,
                    node_id=original.id,
                )
            nodes.append(replace(
                original,
                op=_realization_op(realization),
                inputs=(parameter_layout.parameter_id,),
                attrs={"new_type": original.type},
                metadata={
                    **dict(original.metadata),
                    "introduced_by": (
                        "PropagatePostAutoDistributedFunctionBoundaryLayouts"
                    ),
                    "boundary_layout": "direct_distributed_reshard",
                },
            ))
            nodes.extend(source_helpers.get(original.id, ()))
            continue

        rewritten = replace(
            original,
            inputs=tuple(
                raw_parameter_restores[value].id
                if (
                    value in raw_parameter_restores
                    and original.id in raw_parameter_use_nodes[value]
                )
                else _resolve(value, removed_substitutions)
                for value in original.inputs
            ),
        )
        nodes.append(rewritten)
        nodes.extend(source_helpers.get(original.id, ()))

    functions = []
    for function in module.functions:
        layout = layouts.get(function.name)
        outputs = []
        for index, output_id in enumerate(function.outputs):
            if layout is not None and index in layout.outputs:
                output_layout = layout.outputs[index]
                replacement = output_layout.abi_id
                assert replacement is not None
                if replacement in raw_parameter_restores:
                    replacement = raw_parameter_restores[replacement].id
            elif output_id in raw_parameter_restores:
                replacement = raw_parameter_restores[output_id].id
            else:
                replacement = output_id
            outputs.append(_resolve(replacement, removed_substitutions))
        attrs = dict(function.attrs)
        if layout is not None:
            attrs["post_auto_distributed_boundary_layout"] = {
                "inputs": tuple(sorted(layout.inputs)),
                "outputs": tuple(sorted(layout.outputs)),
            }
        functions.append(replace(
            function,
            outputs=tuple(outputs),
            attrs=attrs,
        ))

    points = tuple(
        replace(point, owner=_resolve(point.owner, removed_owners))
        if point.owner in removed_owners else point
        for point in module.selection_points
    )
    choices = dict(module.metadata.get(
        "distributed_function_boundary_layout_choices", {}))
    for function_name in layouts:
        choices.pop(function_name, None)
    decisions = list(module.metadata.get(
        "distributed_function_boundary_layout_decisions", ()))
    decisions.extend(
        {
            "function": function_name,
            "candidate": layout.candidate_id,
            "solver": "ortools-cp-sat",
        }
        for function_name, layout in sorted(layouts.items())
    )
    metadata = dict(module.metadata)
    if choices:
        metadata["distributed_function_boundary_layout_choices"] = choices
    else:
        metadata.pop("distributed_function_boundary_layout_choices", None)
    metadata["distributed_function_boundary_layout_decisions"] = tuple(decisions)

    return verify_module(replace(
        module,
        nodes=tuple(nodes),
        functions=tuple(functions),
        selection_points=points,
        metadata=metadata,
    ))


def _collect_candidate_layouts(
    module: IRModule,
    realization_policy,
    constants: frozenset[str],
    *,
    users: dict[str, tuple[Node, ...]] | None = None,
    calls: dict[str, tuple[Node, ...]] | None = None,
    enable_caller_output_demand: bool = True,
) -> dict[str, tuple[_FunctionLayout, ...]]:
    users = _user_map(module) if users is None else users
    calls = _calls_by_callee(module) if calls is None else calls
    result = {}
    for function in module.functions:
        function_calls = calls.get(function.name, ())
        if function.name == module.entry or not function_calls:
            continue
        internal = _discover_internal_layout(module, function, users)
        internal = _filter_inapplicable_layout(
            module,
            internal,
            function_calls,
            users,
            realization_policy,
            constants,
        )
        identity = _FunctionLayout(
            function, {}, {}, candidate_id="identity", is_identity=True)
        variants = [identity]
        if internal.inputs or internal.outputs:
            variants.append(internal)
        if enable_caller_output_demand:
            for call in function_calls:
                demanded = _discover_caller_output_demand(
                    module,
                    function,
                    call,
                    users,
                    realization_policy,
                    constants,
                )
                if not demanded:
                    continue
                outputs = dict(internal.outputs)
                for index, output in demanded.items():
                    outputs.setdefault(index, output)
                variant = _FunctionLayout(
                    function,
                    dict(internal.inputs),
                    outputs,
                    candidate_id=f"caller:{call.id}",
                )
                variant = _filter_inapplicable_layout(
                    module,
                    variant,
                    function_calls,
                    users,
                    realization_policy,
                    constants,
                )
                if variant.inputs or variant.outputs:
                    variants.append(variant)
        distinct = []
        keys = set()
        for variant in sorted(
            variants,
            key=lambda value: (not value.is_identity, value.candidate_id),
        ):
            key = _layout_key(variant)
            if key in keys:
                continue
            keys.add(key)
            distinct.append(variant)
        if len(distinct) > 1:
            result[function.name] = tuple(distinct)
    return result


def _discover_internal_layout(
    module: IRModule,
    function: Function,
    users: dict[str, tuple[Node, ...]],
) -> _FunctionLayout:
    owned = {node.id for node in function_nodes(module, function)}
    inputs = {}
    parameter_adapter_ids = set()
    for index, parameter_id in enumerate(function.parameters):
        parameter = module.node_map[parameter_id]
        # The post-distribution extension selects one canonical reshard for an
        # already distributed parameter.  Do not chase a remaining alternate
        # reshard on the next fixed-point iteration: nncase naturally gets
        # this property by excluding DistributedType parameters after its
        # first specialization.
        if parameter.metadata.get("boundary_layout") == "distributed":
            continue
        candidates = tuple(
            node
            for node in users.get(parameter_id, ())
            if (
                node.id in owned
                and node.op in _DISTRIBUTED_ADAPTERS
                and len(node.inputs) == 1
                and isinstance(node.type, DistributedType)
                and node.metadata.get("boundary_layout") not in {
                    "raw_parameter_restore",
                    "distributed_output_abi",
                }
            )
        )
        parameter_adapter_ids.update(node.id for node in candidates)
        if not candidates:
            continue
        groups: dict[DistributedType, list[Node]] = {}
        for candidate in candidates:
            groups.setdefault(candidate.type, []).append(candidate)
        target_type, selected = sorted(
            groups.items(),
            key=lambda item: (-len(item[1]), repr(item[0])),
        )[0]
        if target_type == parameter.type:
            continue
        inputs[index] = _InputLayout(
            index,
            parameter_id,
            parameter.type,
            target_type,
            frozenset(node.id for node in selected),
        )

    outputs = {}
    for index, output_id in enumerate(function.outputs):
        output = module.node_map[output_id]
        if (
            output.id in parameter_adapter_ids
            or output.op not in _DISTRIBUTED_ADAPTERS
            or len(output.inputs) != 1
            or output.metadata.get("boundary_layout") == "distributed_output_abi"
            or users.get(output.id)
        ):
            continue
        source = module.node_map[output.inputs[0]]
        outputs[index] = _OutputLayout(
            index,
            source.id,
            output.type,
            source.type,
            internal_adapter_id=output.id,
        )
    return _FunctionLayout(function, inputs, outputs)


def _discover_caller_output_demand(
    module: IRModule,
    function: Function,
    call: Node,
    users: dict[str, tuple[Node, ...]],
    realization_policy,
    constants: frozenset[str],
) -> dict[int, _OutputLayout]:
    values = {}
    if len(function.outputs) == 1:
        values[0] = call.id
    else:
        for user in users.get(call.id, ()):
            if user.op != "builtin.get_item" or len(user.inputs) != 1:
                continue
            index = int(user.attrs["index"])
            if 0 <= index < len(function.outputs):
                values[index] = user.id

    function_output_roots = {
        output for item in module.functions for output in item.outputs
    }
    outputs = {}
    for index, value_id in values.items():
        value = module.node_map[value_id]
        demands = tuple(
            user
            for user in users.get(value_id, ())
            if user.op in _DISTRIBUTED_ADAPTERS and len(user.inputs) == 1
        )
        if not demands:
            continue
        source = module.node_map[function.outputs[index]]
        for demand in demands:
            if _classify_types(
                source.type,
                demand.type,
                _source_kind(source, constants),
                DistributedReshardUsageKind.INTERNAL,
                realization_policy,
            ) is DistributedReshardRealization.UNSUPPORTED:
                continue
            can_restore = (
                _can_produce_boxing_target(value.type)
                and _classify_types(
                    demand.type,
                    value.type,
                    DistributedReshardSourceKind.INTERNAL,
                    DistributedReshardUsageKind.INTERNAL,
                    realization_policy,
                ) is not DistributedReshardRealization.UNSUPPORTED
            )
            materializing = _is_materializing_collective(
                value.type, demand.type)
            all_match = _all_value_consumers_match(
                value_id,
                demand.type,
                users,
                function_output_roots,
            )
            if not can_restore and not (materializing and all_match):
                continue
            outputs[index] = _OutputLayout(
                index,
                source.id,
                value.type,
                demand.type,
                source_transform=True,
                restore_at_caller=can_restore,
            )
            break
    return outputs


def _filter_inapplicable_layout(
    module: IRModule,
    layout: _FunctionLayout,
    calls: tuple[Node, ...],
    users: dict[str, tuple[Node, ...]],
    realization_policy,
    constants: frozenset[str],
) -> _FunctionLayout:
    owned = {node.id for node in function_nodes(module, layout.function)}
    inputs = {}
    for index, item in layout.inputs.items():
        raw_users = tuple(
            user
            for user in users.get(item.parameter_id, ())
            if user.id in owned and user.op not in _DISTRIBUTED_ADAPTERS
        )
        needs_restore = (
            bool(raw_users)
            or item.parameter_id in layout.function.outputs
            or any(
                output.source_id == item.parameter_id
                for output in layout.outputs.values()
            )
        )
        if needs_restore and (
            not _can_produce_boxing_target(item.source_type)
            or _classify_types(
                item.target_type,
                item.source_type,
                DistributedReshardSourceKind.FUNCTION_PARAMETER,
                DistributedReshardUsageKind.INTERNAL,
                realization_policy,
            ) is DistributedReshardRealization.UNSUPPORTED
        ):
            continue
        for user in users.get(item.parameter_id, ()):
            if (
                user.id not in owned
                or user.op not in _DISTRIBUTED_ADAPTERS
                or user.id in item.adapter_ids
            ):
                continue
            if _classify_types(
                item.target_type,
                user.type,
                DistributedReshardSourceKind.FUNCTION_PARAMETER,
                DistributedReshardUsageKind.INTERNAL,
                realization_policy,
            ) is DistributedReshardRealization.UNSUPPORTED:
                break
        else:
            for call in calls:
                argument = module.node_map[call.inputs[index]]
                if argument.type == item.target_type:
                    continue
                if (
                    argument.op in _DISTRIBUTED_ADAPTERS
                    and len(argument.inputs) == 1
                    and module.node_map[argument.inputs[0]].type == item.target_type
                ):
                    continue
                if _classify(
                    argument,
                    item.target_type,
                    constants,
                    realization_policy,
                    DistributedReshardUsageKind.FUNCTION_BOUNDARY,
                ) is DistributedReshardRealization.UNSUPPORTED:
                    break
            else:
                inputs[index] = item

    outputs = {}
    for index, output in layout.outputs.items():
        if output.source_transform:
            source = module.node_map[output.source_id]
            if _classify(
                source,
                output.abi_type,
                constants,
                realization_policy,
                DistributedReshardUsageKind.INTERNAL,
            ) is DistributedReshardRealization.UNSUPPORTED:
                continue
        if output.restore_at_caller and (
            not _can_produce_boxing_target(output.logical_type)
            or _classify_types(
                output.abi_type,
                output.logical_type,
                DistributedReshardSourceKind.INTERNAL,
                DistributedReshardUsageKind.INTERNAL,
                realization_policy,
            ) is DistributedReshardRealization.UNSUPPORTED
        ):
            continue
        outputs[index] = output
    return replace(layout, inputs=inputs, outputs=outputs)


def _select_layouts(
    module: IRModule,
    candidates: dict[str, tuple[_FunctionLayout, ...]],
    users: dict[str, tuple[Node, ...]],
    calls: dict[str, tuple[Node, ...]],
) -> dict[str, _FunctionLayout]:
    if not candidates:
        return {}
    try:
        from ortools.sat.python import cp_model
    except ImportError as error:  # pragma: no cover - packaging failure.
        raise RuntimeError(
            "Distributed function-boundary planning requires OR-Tools CP-SAT; "
            "install ortools==9.10.4067."
        ) from error

    model = cp_model.CpModel()
    variables = {}
    for function_name, variants in candidates.items():
        choice_vars = []
        for index, _ in enumerate(variants):
            variable = model.NewBoolVar(
                f"distributed_function_layout__"
                f"{_sanitize_name(function_name)}__{index}")
            variables[(function_name, index)] = variable
            choice_vars.append(variable)
        model.AddExactlyOne(choice_vars)

    overrides = module.metadata.get(
        "distributed_function_boundary_layout_choices", {})
    if not isinstance(overrides, Mapping):
        raise IRVerificationError(
            "metadata.distributed_function_boundary_layout_choices must be "
            "a mapping from function name to candidate id.",
            stage=module.stage,
        )
    for function_name, candidate_id in overrides.items():
        variants = candidates.get(str(function_name))
        if variants is None:
            raise IRVerificationError(
                "Distributed function-boundary override references function "
                f"{function_name!r} with no candidates.",
                stage=module.stage,
            )
        matching = [
            index
            for index, variant in enumerate(variants)
            if variant.candidate_id == str(candidate_id)
        ]
        if not matching:
            valid = tuple(variant.candidate_id for variant in variants)
            raise IRVerificationError(
                f"Unknown distributed function-boundary candidate "
                f"{candidate_id!r} for @{function_name}; expected one of "
                f"{valid!r}.",
                stage=module.stage,
            )
        model.Add(variables[(str(function_name), matching[0])] == 1)

    model.Minimize(sum(
        _estimate_variant_cost(
            module,
            variant,
            users,
            calls[function_name],
        ) * variables[(function_name, index)]
        for function_name, variants in candidates.items()
        for index, variant in enumerate(variants)
    ))
    validation = model.Validate()
    if validation:
        raise IRVerificationError(
            f"Distributed function-boundary CP-SAT model is invalid: "
            f"{validation}",
            stage=module.stage,
        )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = _positive_env_float(
        "FLAGMEGA_LAYOUT_SOLVE_MAX_TIME", 10.0)
    solver.parameters.num_search_workers = _positive_env_int(
        "FLAGMEGA_LAYOUT_SOLVE_WORKERS", max((os.cpu_count() or 2) // 2, 1))
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise IRVerificationError(
            "Distributed function-boundary CP-SAT solve failed: "
            f"{solver.StatusName(status)}.",
            stage=module.stage,
        )
    selected = {}
    for function_name, variants in candidates.items():
        variant = next(
            variant
            for index, variant in enumerate(variants)
            if solver.BooleanValue(variables[(function_name, index)])
        )
        if not variant.is_identity:
            selected[function_name] = variant
    return selected


def _estimate_variant_cost(
    module: IRModule,
    layout: _FunctionLayout,
    users: dict[str, tuple[Node, ...]],
    calls: tuple[Node, ...],
) -> int:
    owned = {node.id for node in function_nodes(module, layout.function)}
    transform_count = sum(
        node.id in owned and node.op in _DISTRIBUTED_ADAPTERS
        for node in module.nodes
    )
    call_count = max(len(calls), 1)
    if layout.is_identity:
        return (
            _IDENTITY_VARIANT_PENALTY
            + transform_count * _LOCAL_TRANSFORM_COST_SCALE
        ) * call_count

    removed = sum(len(item.adapter_ids) for item in layout.inputs.values())
    removed += sum(
        output.internal_adapter_id is not None
        for output in layout.outputs.values()
    )
    added = 0
    for item in layout.inputs.values():
        raw_users = any(
            user.id in owned and user.op not in _DISTRIBUTED_ADAPTERS
            for user in users.get(item.parameter_id, ())
        )
        added += (
            raw_users
            or item.parameter_id in layout.function.outputs
            or any(
                output.source_id == item.parameter_id
                for output in layout.outputs.values()
            )
        )
    added += sum(output.source_transform for output in layout.outputs.values())
    local_cost = (
        transform_count - removed + added
    ) * _LOCAL_TRANSFORM_COST_SCALE
    adapter_cost = len(layout.inputs) + sum(
        output.restore_at_caller for output in layout.outputs.values())
    satisfied_demands = 0
    for call in calls:
        values = {0: call.id}
        if len(layout.function.outputs) != 1:
            values = {
                int(user.attrs["index"]): user.id
                for user in users.get(call.id, ())
                if user.op == "builtin.get_item"
            }
        for index, output in layout.outputs.items():
            value_id = values.get(index)
            if value_id is None:
                continue
            satisfied_demands += sum(
                user.op in _DISTRIBUTED_ADAPTERS
                and user.type == output.abi_type
                for user in users.get(value_id, ())
            )
    return (
        (local_cost + adapter_cost) * call_count
        - satisfied_demands * _LOCAL_TRANSFORM_COST_SCALE
    )


def _make_compatibility_view(
    original: Node,
    raw_id: str,
    output: _OutputLayout,
    realization_policy,
) -> Node:
    if not output.restore_at_caller:
        raise IRVerificationError(
            f"Irreversible output {original.id!r} unexpectedly requires a "
            "caller compatibility view.",
            node_id=original.id,
        )
    return _make_adapter_node(
        original.id,
        raw_id,
        output.abi_type,
        output.logical_type,
        DistributedReshardSourceKind.INTERNAL,
        DistributedReshardUsageKind.INTERNAL,
        realization_policy,
        metadata={
            **dict(original.metadata),
            "introduced_by": (
                "PropagatePostAutoDistributedFunctionBoundaryLayouts"
            ),
            "boundary_layout": "logical_view",
        },
    )


def _make_adapter_node(
    node_id: str,
    input_id: str,
    source_type: IRType,
    target_type: IRType,
    source_kind: DistributedReshardSourceKind,
    usage_kind: DistributedReshardUsageKind,
    realization_policy,
    *,
    metadata: Mapping[str, object],
) -> Node:
    realization = _classify_types(
        source_type,
        target_type,
        source_kind,
        usage_kind,
        realization_policy,
    )
    if realization is DistributedReshardRealization.UNSUPPORTED:
        raise IRVerificationError(
            f"Cannot realize distributed boundary {source_type!r} -> "
            f"{target_type!r} for {usage_kind.value}.",
            node_id=node_id,
        )
    return Node(
        node_id,
        _realization_op(realization),
        (input_id,),
        target_type,
        PURE,
        {"new_type": target_type},
        dict(metadata),
    )


def _needs_logical_projection(
    projection: Node,
    output: _OutputLayout,
    users: dict[str, tuple[Node, ...]],
    substitutions: dict[str, str],
    layouts: dict[str, _FunctionLayout],
    function_output_roots: set[str],
) -> bool:
    if not output.restore_at_caller:
        return False
    if projection.id in function_output_roots:
        return True
    for user in users.get(projection.id, ()):
        if user.id in substitutions:
            continue
        if user.op == "builtin.call":
            layout = layouts.get(str(user.attrs["callee"]))
            if layout is not None:
                positions = tuple(
                    index
                    for index, input_id in enumerate(user.inputs)
                    if input_id == projection.id
                )
                if positions and all(
                    index in layout.inputs
                    and layout.inputs[index].target_type == output.abi_type
                    for index in positions
                ):
                    continue
        return True
    return False


def _result_type_after_layout(module: IRModule, layout: _FunctionLayout) -> IRType:
    fields = tuple(
        layout.outputs[index].abi_type
        if index in layout.outputs else module.node_map[output].type
        for index, output in enumerate(layout.function.outputs)
    )
    return fields[0] if len(fields) == 1 else TupleType(fields)


def _all_value_consumers_match(
    value_id: str,
    target_type: IRType,
    users: dict[str, tuple[Node, ...]],
    function_output_roots: set[str],
) -> bool:
    if value_id in function_output_roots:
        return False
    consumers = users.get(value_id, ())
    return bool(consumers) and all(
        consumer.op in _DISTRIBUTED_ADAPTERS
        and len(consumer.inputs) == 1
        and consumer.type == target_type
        for consumer in consumers
    )


def _is_materializing_collective(source_type: IRType, target_type: IRType) -> bool:
    return (
        isinstance(source_type, DistributedType)
        and _has_partial(source_type)
        and _can_produce_boxing_target(target_type)
    )


def _has_partial(value: DistributedType) -> bool:
    return value.partial is not None or any(
        isinstance(policy, SBPPartial) for policy in value.axis_policies)


def _can_produce_boxing_target(value: IRType) -> bool:
    if isinstance(value, TensorType):
        return True
    if isinstance(value, DistributedType):
        return not _has_partial(value)
    if isinstance(value, TupleType):
        return all(_can_produce_boxing_target(field) for field in value.fields)
    return False


def _classify(
    node: Node,
    target_type: IRType,
    constants: frozenset[str],
    realization_policy,
    usage_kind: DistributedReshardUsageKind,
) -> DistributedReshardRealization:
    return _classify_types(
        node.type,
        target_type,
        _source_kind(node, constants),
        usage_kind,
        realization_policy,
    )


def _classify_types(
    source_type: IRType,
    target_type: IRType,
    source_kind: DistributedReshardSourceKind,
    usage_kind: DistributedReshardUsageKind,
    realization_policy,
) -> DistributedReshardRealization:
    if source_type == target_type:
        # This value is only used while planning.  Materialization sites avoid
        # constructing identity adapters, so BOXING is an inert supported tag.
        return DistributedReshardRealization.BOXING
    return realization_policy.classify(DistributedReshardRealizationContext(
        source_type,
        target_type,
        source_kind,
        usage_kind,
    ))


def _source_kind(
    node: Node,
    constants: frozenset[str],
) -> DistributedReshardSourceKind:
    if node.id in constants:
        return DistributedReshardSourceKind.CONSTANT
    if node.op == "builtin.var":
        return DistributedReshardSourceKind.FUNCTION_PARAMETER
    return DistributedReshardSourceKind.INTERNAL


def _realization_op(realization: DistributedReshardRealization) -> str:
    if realization is DistributedReshardRealization.SHARDED_VIEW:
        return "distributed.sharded_view"
    if realization is DistributedReshardRealization.BOXING:
        return "distributed.boxing"
    raise IRVerificationError(
        f"Unsupported distributed reshard realization {realization.value!r}.")


def _calls_by_callee(module: IRModule) -> dict[str, tuple[Node, ...]]:
    live = {
        node.id
        for function in module.functions
        for node in function_nodes(module, function)
    }
    result: dict[str, list[Node]] = {}
    for node in module.nodes:
        if node.id in live and node.op == "builtin.call":
            result.setdefault(str(node.attrs["callee"]), []).append(node)
    return {name: tuple(values) for name, values in result.items()}


def _user_map(module: IRModule) -> dict[str, tuple[Node, ...]]:
    result: dict[str, list[Node]] = {node.id: [] for node in module.nodes}
    for node in module.nodes:
        for input_id in dict.fromkeys(node.inputs):
            result[input_id].append(node)
    return {node_id: tuple(values) for node_id, values in result.items()}


def _layout_key(layout: _FunctionLayout):
    if layout.is_identity:
        return ("identity",)
    return (
        tuple(
            (index, item.source_type, item.target_type)
            for index, item in sorted(layout.inputs.items())
        ),
        tuple(
            (
                index,
                item.logical_type,
                item.abi_type,
                item.internal_adapter_id is not None,
                item.source_transform,
                item.restore_at_caller,
            )
            for index, item in sorted(layout.outputs.items())
        ),
    )


def _resolve(value: str | None, substitutions: dict[str, str]) -> str | None:
    while value in substitutions:
        replacement = substitutions[value]
        if replacement == value:
            break
        value = replacement
    return value


def _fresh_id(stem: str, occupied: set[str]) -> str:
    if stem not in occupied:
        return stem
    index = 1
    while f"{stem}.{index}" in occupied:
        index += 1
    return f"{stem}.{index}"


def _sanitize_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _positive_env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


__all__ = ["propagate_post_auto_distributed_function_boundary_layouts"]
