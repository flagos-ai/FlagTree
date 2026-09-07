# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Canonicalize three packed-QKV RHS values to one TIR ABI value.

This is the FlagMega counterpart of nncase's
``CanonicalizePackedQKVWeightsPass``.  It runs after TIR selection, so graph
packing and AutoDistribution continue to reason about Q/K/V independently.
The pass owns ABI propagation only; target tile sizes and instruction choices
remain in the selected implementation catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from triton.flagmega.ir.tir.kernel_definition import replace_kernel_dispatch, replace_kernel_callables
from math import prod

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import (
    DistributedType,
    Function,
    IRType,
    IRModule,
    Node,
    TensorType,
    get_definition,
    kernel_dispatch_of,
    local_shape,
    logical_type,
)
from triton.flagmega.ir.tir import (
    KernelDispatch,
    PrimFunction,
    PrimParameter,
    Sequential,
)
from triton.flagmega.ir.ops.ntt.packed_qkv_parallel_linear import (
    PackedQKVParallelLinear,
)
from triton.flagmega.ir.types import VectorType
from triton.flagmega.passes.constants import (
    ConstantPhase,
    constant_phase,
    freeze_constant_islands,
    thaw_constant_islands,
)
from triton.flagmega.passes.functions import callee_first_functions, function_nodes


_PACKED_QKV = "ntt.packed_qkv_parallel_linear"
_PACKED_QKV_FUSED_RHS = "ntt.packed_qkv_parallel_linear_fused_rhs"


def _input_index(parameter) -> int:
    if parameter.input_index is None:  # pragma: no cover - decorator invariant.
        raise RuntimeError(f"Unbound ParameterInfo {parameter!r}.")
    return parameter.input_index


_Q_WEIGHT_INDEX = _input_index(PackedQKVParallelLinear.q_weight)
_K_WEIGHT_INDEX = _input_index(PackedQKVParallelLinear.k_weight)
_V_WEIGHT_INDEX = _input_index(PackedQKVParallelLinear.v_weight)
_WEIGHT_INDICES = (_Q_WEIGHT_INDEX, _K_WEIGHT_INDEX, _V_WEIGHT_INDEX)


@dataclass(frozen=True)
class _KernelPlan:
    function: PrimFunction
    fused_type: TensorType
    source_types: tuple[IRType, IRType, IRType]
    projection_n_capacities: tuple[int, int, int]


@dataclass(frozen=True)
class _ParameterGroup:
    q_index: int
    k_index: int
    v_index: int
    source_types: tuple[IRType, IRType, IRType]
    materialization_types: tuple[IRType, IRType, IRType]
    fused_type: TensorType
    fused_parameter: Node

    @property
    def indices(self) -> tuple[int, int, int]:
        return self.q_index, self.k_index, self.v_index


@dataclass(frozen=True)
class _CallGroup:
    q_index: int
    k_index: int
    v_index: int
    source_types: tuple[IRType, IRType, IRType]
    materialization_types: tuple[IRType, IRType, IRType]
    fused_type: TensorType

    @property
    def indices(self) -> tuple[int, int, int]:
        return self.q_index, self.k_index, self.v_index


@dataclass
class _FunctionPlan:
    function: Function
    parameter_groups: dict[tuple[int, int, int], _ParameterGroup]
    call_groups: dict[str, tuple[_CallGroup, ...]]
    allowed_parameter_uses: set[tuple[str, int]]
    parameter_adapters: set[str]
    allowed_adapter_uses: set[tuple[str, int]]


def canonicalize_packed_qkv_weights(module: IRModule) -> IRModule:
    """Rewrite selected packed-QKV calls and propagate their fused RHS ABI.

    Frozen recipe graphs are thawed structurally, extended with the fused
    owner-local materialization, and immediately frozen again. Removed
    optimization decisions are never reconstructed or re-run.
    """

    kernel_plans = _discover_kernel_plans(module)
    if not kernel_plans:
        return module
    refreeze = constant_phase(module) is ConstantPhase.FROZEN
    if refreeze:
        module = thaw_constant_islands(module)
    function_plans, call_owners = _discover_function_plans(module, kernel_plans)
    entry_plan = function_plans[module.entry]
    if entry_plan.parameter_groups:
        raise IRVerificationError(
            "Entry ABI cannot expose unfused packed Q/K/V weight parameters; "
            "they must resolve to compiler-owned constants before TIR selection.",
            stage=module.stage,
        )
    _verify_no_independent_parameter_uses(module, function_plans)

    prim_functions = tuple(
        _rewrite_kernel(function, kernel_plans[function.name])
        if function.name in kernel_plans else function
        for function in module.kernel_callable_map.values()
    )
    removed_parameters = {
        module.functions[index].parameters[parameter_index]
        for index, function in enumerate(module.functions)
        for parameter_index in (
            value
            for group in function_plans[function.name].parameter_groups.values()
            for value in group.indices
        )
    }
    removed_parameter_adapters = {
        adapter
        for plan in function_plans.values()
        for adapter in plan.parameter_adapters
    }
    inserted_parameters = {
        group.fused_parameter.id: group.fused_parameter
        for plan in function_plans.values()
        for group in plan.parameter_groups.values()
    }
    insert_after = {
        plan.function.parameters[group.q_index]: group.fused_parameter
        for plan in function_plans.values()
        for group in plan.parameter_groups.values()
    }

    nodes: list[Node] = []
    fused_values: dict[tuple[str, tuple[str, str, str], TensorType], str] = {}
    for node in module.nodes:
        if node.id in removed_parameters:
            if node.id in insert_after:
                nodes.append(insert_after[node.id])
            continue
        if node.id in removed_parameter_adapters:
            continue
        owner = call_owners.get(node.id)
        if owner is None:
            nodes.append(node)
            continue
        plan = function_plans[owner]
        groups = plan.call_groups[node.id]
        rewritten_inputs, helpers = _rewrite_call_inputs(
            node,
            groups,
            plan,
            module,
            fused_values,
        )
        nodes.extend(helpers)
        nodes.append(replace(node, inputs=rewritten_inputs, metadata={
            **dict(node.metadata),
            "canonicalized_packed_qkv_rhs": True,
        }))

    functions = tuple(
        _rewrite_graph_function(function, function_plans[function.name])
        for function in module.functions
    )
    # Every synthesized parameter must have replaced exactly one original Q
    # parameter.  This catches stale plans before the ordinary verifier sees a
    # less useful missing-node error.
    materialized_ids = {node.id for node in nodes}
    missing = set(inserted_parameters) - materialized_ids
    if missing:  # pragma: no cover - guarded by deterministic insertion above.
        raise IRVerificationError(
            f"Packed-QKV fused parameters were not materialized: {sorted(missing)}.",
            stage=module.stage,
        )
    removed_point_ids = {
        point.id
        for point in module.selection_points
        if point.owner in removed_parameter_adapters
    }
    result = replace_kernel_callables(
        module,
        prim_functions,
        nodes=tuple(nodes),
        functions=functions,
        selection_points=tuple(
            point
            for point in module.selection_points
            if point.id not in removed_point_ids
        ),
        selections=tuple(
            selection
            for selection in module.selections
            if selection.point_id not in removed_point_ids
        ),
    )
    return freeze_constant_islands(result) if refreeze else result


def _discover_kernel_plans(module: IRModule) -> dict[str, _KernelPlan]:
    plans: dict[str, _KernelPlan] = {}
    for function in module.kernel_callable_map.values():
        dispatch = kernel_dispatch_of(function)
        if dispatch is None or dispatch.semantic_op != _PACKED_QKV:
            continue
        input_arity = len(PackedQKVParallelLinear.input_parameters)
        if (
            len(function.runtime_parameters) != input_arity
            or len(dispatch.arguments) != input_arity
        ):
            raise IRVerificationError(
                f"Packed-QKV PrimFunction @{function.name} requires "
                f"{input_arity} input operands.",
                stage=module.stage,
            )
        if str(dispatch.semantic_attrs.get("rhs_layout", "")) != "k_major":
            raise IRVerificationError(
                "Packed-QKV TIR canonicalization requires K-major RHS values.",
                stage=module.stage,
            )
        source_types = tuple(
            function.runtime_parameters[index].type for index in _WEIGHT_INDICES
        )
        fused_type = _fused_weight_type(source_types, function.name)
        if len(function.output_parameters) != 3:
            raise IRVerificationError(
                f"Packed-QKV PrimFunction @{function.name} requires three outputs.",
                stage=module.stage,
            )
        capacities = tuple(
            _projection_n_capacity(parameter.type, function.name)
            for parameter in function.output_parameters
        )
        plans[function.name] = _KernelPlan(
            function,
            fused_type,
            source_types,
            capacities,
        )
    return plans


def _discover_function_plans(module, kernel_plans):
    plans: dict[str, _FunctionPlan] = {}
    call_owners: dict[str, str] = {}
    for function in callee_first_functions(module):
        plan = _FunctionPlan(function, {}, {}, set(), set(), set())
        parameter_indices = {
            parameter: index for index, parameter in enumerate(function.parameters)
        }
        for node in function_nodes(module, function):
            if node.op != "tir.call":
                continue
            callee_name = str(node.attrs.get("callee", ""))
            groups: list[_CallGroup] = []
            kernel_plan = kernel_plans.get(callee_name)
            if kernel_plan is not None:
                groups.append(_CallGroup(
                    _Q_WEIGHT_INDEX,
                    _K_WEIGHT_INDEX,
                    _V_WEIGHT_INDEX,
                    kernel_plan.source_types,
                    kernel_plan.source_types,
                    kernel_plan.fused_type,
                ))
            elif callee_name in plans:
                callee_plan = plans[callee_name]
                groups.extend(
                    _CallGroup(
                        group.q_index,
                        group.k_index,
                        group.v_index,
                        group.source_types,
                        group.materialization_types,
                        group.fused_type,
                    )
                    for group in callee_plan.parameter_groups.values()
                )
            if not groups:
                continue
            plan.call_groups[node.id] = tuple(groups)
            call_owners[node.id] = function.name
            for group in groups:
                _discover_parameter_group(
                    module,
                    plan,
                    node,
                    group,
                    parameter_indices,
                )
        plans[function.name] = plan
    return plans, call_owners


def _discover_parameter_group(module, plan, call, call_group, parameter_indices):
    try:
        actual_ids = tuple(call.inputs[index] for index in call_group.indices)
    except IndexError as error:
        raise IRVerificationError(
            f"Packed-QKV contract call {call.id!r} has an incomplete ABI.",
            stage=module.stage,
            node_id=call.id,
        ) from error
    actual_types = tuple(module.node_map[value].type for value in actual_ids)
    if actual_types != call_group.source_types:
        raise IRVerificationError(
            f"Packed-QKV contract call {call.id!r} has incompatible weight types.",
            stage=module.stage,
            node_id=call.id,
        )
    traced = tuple(
        _trace_parameter_adapter(value, module, parameter_indices)
        for value in actual_ids
    )
    root_ids = tuple(value[0] for value in traced)
    root_types = tuple(module.node_map[value].type for value in root_ids)
    adapter_chains = tuple(value[1] for value in traced)
    parameter_membership = tuple(value in parameter_indices for value in root_ids)
    if not any(parameter_membership):
        return
    if not all(parameter_membership):
        raise IRVerificationError(
            f"Packed-QKV call {call.id!r} mixes parameter and local weight values.",
            stage=module.stage,
            node_id=call.id,
        )
    triple = tuple(parameter_indices[value] for value in root_ids)
    if len(set(triple)) != 3:
        raise IRVerificationError(
            f"Packed-QKV parameter group in @{plan.function.name} contains duplicates.",
            stage=module.stage,
            node_id=call.id,
        )
    existing = plan.parameter_groups.get(triple)
    if existing is None:
        claimed = {
            index
            for group in plan.parameter_groups.values()
            for index in group.indices
        }
        if claimed.intersection(triple):
            raise IRVerificationError(
                f"Packed-QKV parameter groups overlap in @{plan.function.name}.",
                stage=module.stage,
                node_id=call.id,
            )
        q_parameter = module.node_map[root_ids[0]]
        fused_id = f"{q_parameter.id}.qkv_fused"
        fused_parameter = Node(
            fused_id,
            "builtin.var",
            (),
            call_group.fused_type,
            attrs={"name": f"{q_parameter.attrs['name']}_qkv_fused"},
            metadata={
                **dict(q_parameter.metadata),
                "canonicalized_from": root_ids,
                "packed_layout": "k_major",
            },
        )
        plan.parameter_groups[triple] = _ParameterGroup(
            *triple,
            root_types,
            call_group.materialization_types,
            call_group.fused_type,
            fused_parameter,
        )
    elif (
        existing.source_types != root_types
        or existing.materialization_types != call_group.materialization_types
        or existing.fused_type != call_group.fused_type
    ):
        raise IRVerificationError(
            f"Packed-QKV parameter group {triple} in @{plan.function.name} "
            "has incompatible layouts.",
            stage=module.stage,
            node_id=call.id,
        )
    for input_index, adapters in zip(call_group.indices, adapter_chains):
        if not adapters:
            plan.allowed_parameter_uses.add((call.id, input_index))
            continue
        plan.parameter_adapters.update(adapters)
        plan.allowed_parameter_uses.add((adapters[-1], 0))
        plan.allowed_adapter_uses.add((call.id, input_index))
        plan.allowed_adapter_uses.update(
            (outer, 0) for outer in adapters[:-1]
        )


def _trace_parameter_adapter(actual_id, module, parameter_indices):
    current = actual_id
    adapters: list[str] = []
    while current not in parameter_indices:
        node = module.node_map[current]
        if not _is_parameter_adapter(node, module):
            break
        adapters.append(node.id)
        current = node.inputs[0]
    return current, tuple(adapters)


def _is_parameter_adapter(node, module) -> bool:
    """Recognize ABI-only views/boxing that fused-RHS materialization absorbs.

    AutoDistribution is allowed to lower a function-parameter reshard to a
    selected ``tir.call`` before this pre-bufferize ABI pass.  Such a call is
    still an adapter, not an independent packed-weight producer: the fused
    owner-major readonly-data recipe at the outermost caller performs the same
    source-to-selected-layout mapping for all three weights at once.  Only the
    typed, unary ``distributed.boxing`` contract is admitted here; arbitrary
    TIR calls must remain visible and therefore prevent parameter fusion.
    """

    if len(node.inputs) != 1:
        return False
    if node.op == "distributed.sharded_view":
        return True
    if node.op != "tir.call":
        return False
    callee = module.kernel_callable_map.get(str(node.attrs.get("callee", "")))
    if callee is None:
        return False
    dispatch = kernel_dispatch_of(callee)
    if (
        dispatch is None
        or dispatch.semantic_op != "distributed.boxing"
        or len(callee.runtime_parameters) != 1
    ):
        return False
    source = module.node_map[node.inputs[0]]
    return (
        source.type == callee.runtime_parameter_types[0]
        and node.type == callee.runtime_return_type
        and logical_type(source.type) == logical_type(node.type)
    )


def _verify_no_independent_parameter_uses(module, plans):
    for plan in plans.values():
        parameter_ids = {
            plan.function.parameters[index]
            for group in plan.parameter_groups.values()
            for index in group.indices
        }
        for output in plan.function.outputs:
            if output in parameter_ids:
                raise IRVerificationError(
                    f"Packed-QKV weight parameter {output!r} in @{plan.function.name} "
                    "has an independent reference as a function result.",
                    stage=module.stage,
                    node_id=output,
                )
        for node in function_nodes(module, plan.function):
            for index, input_id in enumerate(node.inputs):
                if (
                    input_id in parameter_ids
                    and (node.id, index) not in plan.allowed_parameter_uses
                ):
                    raise IRVerificationError(
                        f"Packed-QKV weight parameter {input_id!r} in "
                        f"@{plan.function.name} has an independent reference.",
                        stage=module.stage,
                        node_id=node.id,
                    )
                if (
                    input_id in plan.parameter_adapters
                    and (node.id, index) not in plan.allowed_adapter_uses
                ):
                    raise IRVerificationError(
                        f"Packed-QKV parameter adapter {input_id!r} in "
                        f"@{plan.function.name} has an independent reference.",
                        stage=module.stage,
                        node_id=node.id,
                    )


def _rewrite_kernel(function, plan):
    dispatch = kernel_dispatch_of(function)
    assert dispatch is not None
    q_parameter = function.runtime_parameters[_Q_WEIGHT_INDEX]
    fused_parameter = PrimParameter(
        f"{q_parameter.name}_qkv_fused",
        plan.fused_type,
        q_parameter.role,
    )
    parameters = []
    for index, parameter in enumerate(function.runtime_parameters):
        if index == _Q_WEIGHT_INDEX:
            parameters.append(fused_parameter)
        elif index not in {_K_WEIGHT_INDEX, _V_WEIGHT_INDEX}:
            parameters.append(parameter)
    parameters.extend(function.output_parameters)
    parameters.extend(function.workspaces)
    argument_names = []
    for index, name in enumerate(dispatch.arguments):
        if index == _Q_WEIGHT_INDEX:
            argument_names.append(fused_parameter.name)
        elif index not in {_K_WEIGHT_INDEX, _V_WEIGHT_INDEX}:
            argument_names.append(name)
    removed_names = {dispatch.arguments[index] for index in _WEIGHT_INDICES}
    reads = tuple(
        fused_parameter.name
        if value == dispatch.arguments[_Q_WEIGHT_INDEX]
        else value
        for value in dispatch.reads
        if value not in removed_names
        or value == dispatch.arguments[_Q_WEIGHT_INDEX]
    )
    if dispatch.memory_effects:
        fused_effect = next(
            effect
            for name, effect in dispatch.memory_effects
            if name == dispatch.arguments[_Q_WEIGHT_INDEX]
        )
        memory_effects = tuple(
            (fused_parameter.name, fused_effect)
            if name == dispatch.arguments[_Q_WEIGHT_INDEX]
            else (name, effect)
            for name, effect in dispatch.memory_effects
            if name not in removed_names
            or name == dispatch.arguments[_Q_WEIGHT_INDEX]
        )
    else:
        memory_effects = ()
    semantic_parameters = _rewrite_distribution_contract(
        dispatch.semantic_parameters,
        function,
        plan,
        owner="semantic TIR selection",
    )
    microkernel = dispatch.microkernel
    if microkernel is not None:
        microkernel = replace(
            microkernel,
            parameters=_rewrite_distribution_contract(
                microkernel.parameters,
                function,
                plan,
                owner="TIR microkernel selection",
            ),
        )
    rewritten = replace(
        dispatch,
        semantic_op=_PACKED_QKV_FUSED_RHS,
        arguments=tuple(argument_names),
        inplace_alias_candidates=(),
        semantic_parameters=semantic_parameters,
        semantic_attrs={
            **dict(dispatch.semantic_attrs),
            "projection_n_capacities": plan.projection_n_capacities,
        },
        microkernel=microkernel,
        reads=reads,
        memory_effects=memory_effects,
    )
    return replace_kernel_dispatch(
        function,
        rewritten,
        parameters=tuple(parameters),
    )


def _rewrite_distribution_contract(parameters, function, plan, *, owner):
    """Keep the selected distributed ABI atomic with the PrimFunction ABI."""

    result = dict(parameters)
    distribution = result.get("distribution")
    if not hasattr(distribution, "get"):
        return result
    input_types = tuple(distribution.get(
        "input_types", function.runtime_parameter_types
    ))
    if len(input_types) != len(function.runtime_parameters):
        raise IRVerificationError(
            f"Packed-QKV {owner} has a stale distributed input ABI.",
        )
    if tuple(input_types[index] for index in _WEIGHT_INDICES) != plan.source_types:
        raise IRVerificationError(
            f"Packed-QKV {owner} distributed RHS types disagree with its "
            "PrimFunction ABI.",
        )
    rewritten_inputs = []
    for index, value_type in enumerate(input_types):
        if index == _Q_WEIGHT_INDEX:
            rewritten_inputs.append(plan.fused_type)
        elif index not in {_K_WEIGHT_INDEX, _V_WEIGHT_INDEX}:
            rewritten_inputs.append(value_type)
    result["distribution"] = {
        **dict(distribution),
        "input_types": tuple(rewritten_inputs),
    }
    return result


def _rewrite_graph_function(function, plan):
    by_q = {group.q_index: group for group in plan.parameter_groups.values()}
    removed = {
        index
        for group in plan.parameter_groups.values()
        for index in (group.k_index, group.v_index)
    }
    parameters = []
    for index, parameter in enumerate(function.parameters):
        if index in removed:
            continue
        group = by_q.get(index)
        parameters.append(group.fused_parameter.id if group is not None else parameter)
    return replace(function, parameters=tuple(parameters))


def _rewrite_call_inputs(node, groups, plan, module, fused_values):
    by_q = {group.q_index: group for group in groups}
    removed = {
        index for group in groups for index in (group.k_index, group.v_index)
    }
    parameter_groups = {
        group.indices: group for group in plan.parameter_groups.values()
    }
    parameter_indices = {
        parameter: index for index, parameter in enumerate(plan.function.parameters)
    }
    helpers: list[Node] = []
    inputs: list[str] = []
    for index, input_id in enumerate(node.inputs):
        if index in removed:
            continue
        group = by_q.get(index)
        if group is None:
            inputs.append(input_id)
            continue
        actual_ids = tuple(node.inputs[value] for value in group.indices)
        actual_root_ids = tuple(
            _trace_parameter_adapter(value, module, parameter_indices)[0]
            for value in actual_ids
        )
        actual_parameter_indices = tuple(
            parameter_indices.get(value, -1) for value in actual_root_ids
        )
        parameter_group = parameter_groups.get(actual_parameter_indices)
        if parameter_group is not None:
            inputs.append(parameter_group.fused_parameter.id)
            continue
        cache_key = (plan.function.name, actual_ids, group.fused_type)
        if cache_key in fused_values:
            inputs.append(fused_values[cache_key])
            continue
        suffix = "qkv_fused" if len(groups) == 1 else f"qkv_fused_{index}"
        actual_nodes = tuple(module.node_map[value] for value in actual_ids)
        concat_inputs = actual_nodes
        concat_axis = 1
        distributed_materialization = tuple(
            isinstance(value, DistributedType)
            for value in group.materialization_types
        )
        if any(distributed_materialization) and not all(distributed_materialization):
            raise IRVerificationError(
                f"Packed-QKV call {node.id!r} has mixed materialization domains.",
                stage=module.stage,
                node_id=node.id,
            )
        if all(distributed_materialization):
            materialize = get_definition("distributed.materialize_local_shards")
            local_values = []
            for ordinal, (source, desired_type) in enumerate(zip(
                actual_nodes, group.materialization_types
            )):
                assert isinstance(desired_type, DistributedType)
                materialize_source = source
                if source.type != desired_type:
                    if logical_type(source.type) != logical_type(desired_type):
                        raise IRVerificationError(
                            f"Packed-QKV call {node.id!r} cannot view RHS "
                            f"{ordinal} as its selected distributed type.",
                            stage=module.stage,
                            node_id=node.id,
                        )
                    view = get_definition("distributed.sharded_view")
                    viewed = view.prepare(
                        (source,), {"new_type": desired_type}
                    )
                    materialize_source = Node(
                        f"{node.id}.{suffix}.view_{ordinal}",
                        "distributed.sharded_view",
                        (source.id,),
                        viewed.result_type,
                        viewed.effect,
                        viewed.attrs,
                        {
                            "canonicalized_from": source.id,
                            "packed_layout": "selected_sbp_view",
                        },
                    )
                    helpers.append(materialize_source)
                prepared = materialize.prepare((materialize_source,), {})
                helper = Node(
                    f"{node.id}.{suffix}.source_{ordinal}",
                    "distributed.materialize_local_shards",
                    (materialize_source.id,),
                    prepared.result_type,
                    prepared.effect,
                    prepared.attrs,
                    {
                        "canonicalized_from": source.id,
                        "packed_layout": "owner_major_local_shards",
                    },
                )
                helpers.append(helper)
                local_values.append(helper)
            concat_inputs = tuple(local_values)
            concat_axis = 2
        definition = get_definition("tensors.concat")
        prepared = definition.prepare(concat_inputs, {"axis": concat_axis})
        if prepared.result_type != group.fused_type:
            raise IRVerificationError(
                f"Packed-QKV fused RHS type disagrees at call {node.id!r}.",
                stage=module.stage,
                node_id=node.id,
            )
        helper = Node(
            f"{node.id}.{suffix}",
            "tensors.concat",
            tuple(value.id for value in concat_inputs),
            prepared.result_type,
            prepared.effect,
            prepared.attrs,
            {
                "canonicalized_from": actual_ids,
                "packed_layout": "k_major",
            },
        )
        helpers.append(helper)
        fused_values[cache_key] = helper.id
        inputs.append(helper.id)
    return tuple(inputs), tuple(helpers)


def _plain_tensor(value_type, owner):
    value = logical_type(value_type)
    if not isinstance(value, TensorType):
        raise IRVerificationError(
            f"Packed-QKV {owner} weight must be tensor-like, got "
            f"{type(value_type).__name__}."
        )
    return value


def _fused_weight_type(source_types, owner):
    logical_types = tuple(_plain_tensor(value, owner) for value in source_types)
    if any(value.rank != 2 for value in logical_types):
        raise IRVerificationError(
            f"Packed-QKV {owner} weights must be rank 2."
        )
    first = logical_types[0]
    if any(
        value.dtype != first.dtype
        or value.layout != first.layout
        or value.shape[0] != first.shape[0]
        for value in logical_types[1:]
    ):
        raise IRVerificationError(
            f"Packed-QKV {owner} weights require one dtype, layout, and K capacity."
        )
    distributed = tuple(isinstance(value, DistributedType) for value in source_types)
    if any(distributed) and not all(distributed):
        raise IRVerificationError(
            f"Packed-QKV {owner} cannot mix distributed and local weights."
        )
    if all(distributed):
        values = tuple(value for value in source_types if isinstance(value, DistributedType))
        if any(
            value.placement != values[0].placement or value.partial is not None
            for value in values
        ):
            raise IRVerificationError(
                f"Packed-QKV {owner} distributed weights require one placement without partial values."
            )
        local_shapes = tuple(local_shape(value) for value in values)
        if any(
            len(shape) != 2
            or shape[0] != local_shapes[0][0]
            or not shape[1].is_fixed
            for shape in local_shapes
        ):
            raise IRVerificationError(
                f"Packed-QKV {owner} distributed weights require compatible fixed local capacities."
            )
        owners = prod(values[0].placement.hierarchy)
        return TensorType(
            first.dtype,
            (
                owners,
                local_shapes[0][0],
                sum(shape[1].fixed_value for shape in local_shapes),
            ),
            first.layout,
        )
    if any(not value.shape[1].is_fixed for value in logical_types):
        raise IRVerificationError(
            f"Packed-QKV {owner} weights require fixed projection capacities."
        )
    return TensorType(
        first.dtype,
        (first.shape[0], sum(value.shape[1].fixed_value for value in logical_types)),
        first.layout,
    )


def _projection_n_capacity(value_type, owner):
    value = logical_type(value_type)
    if not isinstance(value, TensorType) or value.rank < 1 or not value.shape[-1].is_fixed:
        raise IRVerificationError(
            f"Packed-QKV @{owner} outputs require fixed packed N capacity."
        )
    lanes = prod(value.dtype.lanes) if isinstance(value.dtype, VectorType) else 1
    return value.shape[-1].fixed_value * lanes


@dataclass(frozen=True)
class CanonicalizePackedQKVWeightsPass:
    name: str = "CanonicalizePackedQKVWeights"
    preserves: frozenset[str] = frozenset()

    def run(self, module: IRModule) -> IRModule:
        return canonicalize_packed_qkv_weights(module)


__all__ = [
    "CanonicalizePackedQKVWeightsPass",
    "canonicalize_packed_qkv_weights",
]
