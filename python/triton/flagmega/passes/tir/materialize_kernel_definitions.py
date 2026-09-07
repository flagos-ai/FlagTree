# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Intern selected kernel contracts without executable function boundaries."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.model import (
    IRModule,
    Node,
    RefType,
    TensorType,
    TupleType,
    logical_type,
)
from triton.flagmega.ir.memory_effect import MemoryAccessMode, MemoryEffect
from triton.flagmega.ir.ops.core import ParameterInfo, get_definition
from triton.flagmega.ir.tir import (
    KernelDispatch,
    InplaceAliasCandidate,
    PrimFunction,
    KernelDefinition,
    PrimParameter,
    PrimParameterRole,
    Return,
    ReturnBinding,
    Sequential,
    ValueRef,
    WorkspaceRequirement,
    kernel_dispatch_of,
)


def materialize_kernel_definitions(module: IRModule) -> IRModule:
    """Intern one typed kernel contract per implementation/physical ABI.

    Graph references share immutable selection data. Later KernelInvoke nodes
    bind actual buffers in the enclosing function's scheduling region; these
    definitions are not executable functions and own no pipeline lifetime.
    """

    identities: dict[str, KernelDefinition] = {}
    names: dict[str, str] = {}
    result_functions = list(module.kernel_callable_map.values())
    occupied = {value.name for value in module.functions}
    for function in module.kernel_callable_map.values():
        occupied.add(function.name)
        identity = _existing_kernel_identity(function)
        if identity is not None:
            identities[identity] = function
            names[function.name] = identity

    nodes: list[Node] = []
    for node in module.nodes:
        if node.op != "tir.kernel":
            nodes.append(node)
            continue
        function, identity = _kernel_definition(node, module)
        existing = identities.get(identity)
        if existing is None:
            name = function.name
            if name in occupied:
                previous = names.get(name)
                if previous != identity:
                    raise IRVerificationError(
                        f"Generated PrimFunction @{name} collides with an unrelated function.",
                        stage=module.stage,
                        node_id=node.id,
                    )
            else:
                occupied.add(name)
                names[name] = identity
                identities[identity] = function
                result_functions.append(function)
            existing = function
        nodes.append(Node(
            id=node.id,
            op="tir.call",
            inputs=node.inputs,
            type=node.type,
            effect=node.effect,
            attrs={"callee": existing.name},
            metadata=node.metadata,
        ))
    from triton.flagmega.ir.tir.kernel_definition import replace_kernel_callables
    return replace_kernel_callables(module, result_functions, nodes=tuple(nodes))


def _kernel_definition(node: Node, module: IRModule) -> tuple[KernelDefinition, str]:
    definition = get_definition(str(node.attrs["semantic_op"]))
    parameter_infos = tuple(_parameter_info(definition.input_parameters, index) for index in range(len(node.inputs)))
    parameter_effects = tuple(_physical_memory_effect(value) for value in parameter_infos)
    parameter_names = _unique_parameter_names(parameter_infos)
    roles = tuple(
        _parameter_role(effect, module.node_map[input_id].type)
        for effect, input_id in zip(
            parameter_effects, node.inputs, strict=True
        )
    )
    parameters = tuple(
        PrimParameter(name, value_type, role)
        for name, value_type, role in zip(
            parameter_names,
            (module.node_map[value].type for value in node.inputs),
            roles,
        )
    )
    reads = tuple(
        name for name, effect in zip(parameter_names, parameter_effects)
        if effect.physical_mode & MemoryAccessMode.READ
    )
    result_fields = _result_fields(node.type)
    result_names = tuple(name for name, _ in result_fields)
    result_effects = definition.result_memory_effects or tuple(
        _result_memory_effect(value_type) for _, value_type in result_fields
    )
    if len(result_effects) != len(result_fields):
        raise IRVerificationError(
            f"Op {definition.op_name!r} declares {len(result_effects)} result "
            f"memory effects for {len(result_fields)} ABI results.",
            stage=module.stage, node_id=node.id,
        )
    if any(
        isinstance(logical_type(value_type), RefType) and effect != MemoryEffect.NONE
        for (_, value_type), effect in zip(result_fields, result_effects)
    ):
        raise IRVerificationError(
            "An input-owned Ref result must not declare a second physical access.",
            stage=module.stage, node_id=node.id,
        )
    reads += tuple(
        name for name, effect in zip(result_names, result_effects)
        if effect.physical_mode & MemoryAccessMode.READ
    )
    writes = tuple(
        name for name, effect in zip(parameter_names, parameter_effects)
        if effect.physical_mode & MemoryAccessMode.WRITE
    ) + tuple(
        name for name, effect in zip(result_names, result_effects)
        if effect.physical_mode & MemoryAccessMode.WRITE
    )
    memory_effects = tuple(
        (name, effect)
        for name, effect in zip(parameter_names, parameter_effects)
    ) + tuple(
        (name, effect)
        for name, effect in zip(result_names, result_effects)
    )
    kernel_parameters = dict(node.attrs.get("parameters", {}))
    raw_workspaces = tuple(kernel_parameters.pop("workspaces", ()))
    workspaces = tuple(_workspace_requirement(value) for value in raw_workspaces)
    common_dispatch = {
        "semantic_op": str(node.attrs["semantic_op"]),
        "arguments": parameter_names,
        "outputs": result_names,
        "workspaces": workspaces,
        "inplace_alias_candidates": _inplace_alias_candidates(
            definition,
            parameter_infos,
            parameter_names,
            node.type,
            result_names,
        ),
        "semantic_attrs": node.attrs["semantic_attrs"],
        "reads": reads,
        "writes": writes,
        "memory_effects": memory_effects,
        "effect_kind": node.effect.kind.value,
        "effect_resource": node.effect.resource,
    }
    if "semantic_candidate" in node.attrs:
        dispatch = KernelDispatch(
            **common_dispatch,
            semantic_candidate=str(node.attrs["semantic_candidate"]),
            semantic_parameters=node.attrs.get("semantic_parameters", {}),
            semantic_facts=node.attrs.get("semantic_facts", {}),
        )
    else:
        dispatch = KernelDispatch(
            **common_dispatch,
            candidate=str(node.attrs["candidate"]),
            parameters=kernel_parameters,
            facts=node.attrs["facts"],
        )
    payload = {
        "dispatch": dispatch.to_data(),
        "input_types": [module.node_map[value].type.to_data() for value in node.inputs],
        "result_type": node.type.to_data(),
        "effect": node.effect.to_data(),
    }
    if dispatch.semantic_op in {
        "distributed.boxing",
        "distributed.force_boxing",
    }:
        # nncase lowers Boxing recursively at its call site.  Its physical
        # ABI depends not only on DistributedType but also on whether the
        # selected source/result buffer is canonical-global, compact-local,
        # or compact-per-owner.  Those storage facts are assigned by the
        # later buffer planner and are intentionally absent from graph types,
        # so two Boxing sites must not share a pre-bufferization PrimFunction.
        payload["call_site"] = node.id
    identity = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    if "semantic_candidate" in node.attrs:
        family = dispatch.semantic_op
        variant = dispatch.semantic_candidate
    else:
        kernel_parameters = node.attrs["parameters"]
        family = str(kernel_parameters.get("family", dispatch.semantic_op))
        variant = str(kernel_parameters.get("variant", dispatch.candidate))
    stem = re.sub(r"[^a-zA-Z0-9_]+", "_", f"{family}_{variant}").strip("_").lower()
    name = f"kernel_{stem[:64]}_{identity[:16]}"
    result_parameters = tuple(
        PrimParameter(name, value_type, PrimParameterRole.OUTPUT)
        for name, value_type in result_fields
    )
    workspace_parameters = tuple(
        PrimParameter(
            value.name,
            value.type,
            PrimParameterRole.WORKSPACE,
            memory_space=value.memory_space,
        )
        for value in workspaces
    )
    function = KernelDefinition(
        name=name,
        module_kind="triton",
        parameters=(*parameters, *result_parameters, *workspace_parameters),
        dispatch=dispatch,
        results=Return(tuple(
            ReturnBinding(ValueRef(name, value_type), name)
            for name, value_type in result_fields
        )),
        attrs={},
    )
    return function, identity


def _parameter_role(effect: MemoryEffect, value_type) -> PrimParameterRole:
    if effect.physical_mode & MemoryAccessMode.WRITE:
        return PrimParameterRole.INOUT
    if effect.physical_mode & MemoryAccessMode.READ:
        return PrimParameterRole.INPUT
    logical = logical_type(value_type)
    # Rank-zero tensor operands are immediate scalar values in the current
    # TIR ABI.  They do not access memory according to ParameterInfo, but the
    # renderer still consumes their value (for example a layer id or boolean
    # control).  Non-scalar NONE-effect operands carry only static type/shape.
    if isinstance(logical, TensorType) and not logical.shape:
        return PrimParameterRole.INPUT
    return PrimParameterRole.METADATA


def _result_fields(result_type) -> tuple[tuple[str, object], ...]:
    """Expose every top-level tuple field as a distinct kernel ABI result.

    Graph ``tir.call`` retains its aggregate :class:`TupleType`; the
    PrimFunction ABI names the physical results independently so kernel
    dispatch, bufferization, dumps, and code generation can reason about each
    value without an out-of-band workspace convention.
    """

    if isinstance(result_type, TupleType):
        if not result_type.fields:
            raise IRVerificationError("Selected kernel tuple result cannot be empty.")
        return tuple(
            (f"result_{index}", value_type)
            for index, value_type in enumerate(result_type.fields)
        )
    return (("result", result_type),)


def _existing_kernel_identity(function: PrimFunction) -> str | None:
    dispatch = kernel_dispatch_of(function)
    if dispatch is None:
        return None
    payload = {
        "dispatch": dispatch.to_data(),
        "input_types": [value.type.to_data() for value in function.runtime_parameters],
        "result_type": function.runtime_return_type.to_data(),
        "effect": {
            "kind": dispatch.effect_kind,
            "resource": dispatch.effect_resource,
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _workspace_requirement(value) -> WorkspaceRequirement:
    if isinstance(value, WorkspaceRequirement):
        return value
    if not hasattr(value, "get"):
        raise IRVerificationError("Kernel workspace requirements must be mappings.")
    return WorkspaceRequirement(
        name=str(value.get("name", "")),
        type=value.get("type"),
        memory_space=str(value.get("memory_space", "workspace")),
        alignment=int(value.get("alignment", 256)),
        lifetime=str(value.get("lifetime", "invocation")),
    )


def _parameter_info(parameters: tuple[ParameterInfo, ...], index: int) -> ParameterInfo:
    for parameter in parameters:
        assert parameter.input_index is not None
        if parameter.variadic and index >= parameter.input_index:
            return parameter
        if parameter.input_index == index:
            return parameter
    raise IRVerificationError(f"Selected kernel operand {index} has no ParameterInfo.")


def _physical_memory_effect(parameter: ParameterInfo) -> MemoryEffect:
    """Return the op-local effect declared by its ParameterInfo schema."""

    return parameter.memory_effect


def _result_memory_effect(value_type) -> MemoryEffect:
    """Describe physical writes performed through an implicit result.

    A Ref result is only the SSA identity returned for an input-owned resource;
    the corresponding inout operand already declares the actual access.  It is
    not a second whole-resource write.  Tensor results own output storage and
    therefore retain the ordinary write contract.
    """

    return (
        MemoryEffect.NONE
        if isinstance(logical_type(value_type), RefType)
        else MemoryEffect.WRITE
    )


def _unique_parameter_names(parameters: tuple[ParameterInfo, ...]) -> tuple[str, ...]:
    totals: dict[str, int] = {}
    result = []
    for parameter in parameters:
        ordinal = totals.get(parameter.name, 0)
        totals[parameter.name] = ordinal + 1
        result.append(parameter.name if ordinal == 0 else f"{parameter.name}_{ordinal}")
    return tuple(result)


def _inplace_alias_candidates(
    definition,
    parameter_infos: tuple[ParameterInfo, ...],
    parameter_names: tuple[str, ...],
    result_type,
    result_names: tuple[str, ...],
) -> tuple[InplaceAliasCandidate, ...]:
    """Freeze definition-level ParameterInfo aliases into the named TIR ABI."""

    candidates: list[InplaceAliasCandidate] = []

    def add(output_name: str, parameter: ParameterInfo) -> None:
        for actual, name in zip(parameter_infos, parameter_names):
            if actual is parameter:
                candidates.append(
                    InplaceAliasCandidate(output=output_name, input=name)
                )

    if isinstance(result_type, TupleType):
        for index, parameter in enumerate(definition.inplace_output_parameters):
            if parameter is not None and index < len(result_names):
                add(result_names[index], parameter)
    elif result_names:
        for parameter in definition.inplace_input_parameters:
            add(result_names[0], parameter)
    return tuple(dict.fromkeys(candidates))


@dataclass(frozen=True)
class MaterializeKernelDefinitionsPass:
    name: str = "MaterializeKernelDefinitions"
    preserves: frozenset[str] = frozenset()

    def run(self, module: IRModule) -> IRModule:
        return materialize_kernel_definitions(module)


__all__ = ["MaterializeKernelDefinitionsPass", "materialize_kernel_definitions"]
