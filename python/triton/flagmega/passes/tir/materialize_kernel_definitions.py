# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Intern selected kernel contracts without executable function boundaries."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.memory_effect import (
    MemoryAccessMode,
    MemoryEffect,
    implementation_memory_effect,
)
from triton.flagmega.ir.model import (
    IRModule,
    Node,
    RefType,
    TensorType,
    TupleType,
    logical_type,
    tensor_type,
)
from triton.flagmega.ir.ops.core import ParameterInfo, get_definition
from triton.flagmega.ir.tir import (
    InplaceAliasCandidate,
    KernelDefinition,
    KernelDispatch,
    PrimFunction,
    PrimParameter,
    PrimParameterRole,
    Return,
    ReturnBinding,
    ValueRef,
    WorkspaceRequirement,
    kernel_dispatch_of,
)


def materialize_kernel_definitions(module: IRModule, *, module_kind: str = "triton") -> IRModule:
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
        function, identity = _kernel_definition(node, module, module_kind=module_kind)
        if function.dispatch.parameters.get("variant") in ("r004_reference_v4", "r004_column_v4", "r004_column_v6", 'r017_w1', 'r017_w2', 'r017_w4', 'r017_w8', 'r017_w16', 'r017_w32', 'r017_w64', 'r017_split', 'r017_split4', 'r017_split8', 'r017_split4lut'):
            # One coarse call retains the tuple/state effect. Named native
            # definitions are contracts; the package projects the 12-entry conditional registry.
            for phase_definition in r004_phase_definitions(function):
                previous = next((f for f in result_functions
                                 if f.name == phase_definition.name), None)
                if previous is not None and previous != phase_definition:
                    raise IRVerificationError("Conflicting frozen r004 phase definition.")
                if previous is None:
                    result_functions.append(phase_definition)
                    occupied.add(phase_definition.name)
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


def _q8_formal_alignment(node, name, *, result=False):
    """Declared scalar access requirements propagate through caller aliases."""
    from triton.flagmega.ir.ops.math._iq4_q8 import OPS
    op=node.attrs.get("semantic_op")
    if op not in OPS: return None
    from triton.flagmega.codegen.triton.iq4_q8_renderer import parameter_alignments
    if node.attrs.get('parameters', {}).get('variant') in ('q8_r006_consumer_barrier_v3','r038_glu_global_prefetch_v1','r006_glu_persistent_v1','r006_glu_transport_t1','r006_glu_transport_t2','r006_glu_transport_t3') and name in ('weight','gate_weight','up_weight'):
        return 16
    if node.attrs.get("parameters", {}).get("variant") in ("r046_glu_staging_v1", "r047_glu_occupancy_v1", "r048_glu_larger_tiles_v1") and name in ("gate_weight", "up_weight"):
        return 16
    return parameter_alignments(OPS[op],node.attrs["semantic_attrs"])[name]



def _kernel_definition(node: Node, module: IRModule, *, module_kind: str = "triton") -> tuple[KernelDefinition, str]:
    definition = get_definition(str(node.attrs["semantic_op"]))
    parameter_infos = tuple(_parameter_info(definition.input_parameters, index) for index in range(len(node.inputs)))
    parameter_names = _unique_parameter_names(parameter_infos)
    parameter_effects = tuple(
        _implementation_memory_effect(_physical_memory_effect(value), node, module,
                                      parameter_name=name, input_names=parameter_names)
        for value, name in zip(parameter_infos, parameter_names, strict=True)
    )
    roles = tuple(
        _parameter_role(effect, module.node_map[input_id].type)
        for effect, input_id in zip(
            parameter_effects, node.inputs, strict=True
        )
    )
    from triton.flagmega.ir.ops.math._iq4_q8 import OPS as q8_ops
    if definition.op_name in q8_ops:
        # Keep placeholder pointer identity in the formal ABI without READ effects.
        roles = tuple(PrimParameterRole.INPUT if name.endswith("_scale") else role
                      for name, role in zip(parameter_names, roles, strict=True))
    parameters = tuple(
        PrimParameter(name, value_type, role, alignment_bytes=_q8_formal_alignment(node, name))
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
    result_effects = tuple(
        _implementation_memory_effect(effect, node, module, is_result=True, input_names=parameter_names)
        for effect in result_effects
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
    if kernel_parameters.get("variant") in ("r004_reference_v4", "r004_column_v4", "r004_column_v6", 'r017_w1', 'r017_w2', 'r017_w4', 'r017_w8', 'r017_w16', 'r017_w32', 'r017_w64', 'r017_split', 'r017_split4', 'r017_split8', 'r017_split4lut'):
        from triton.flagmega.rules.ntt.r004_gdn_phase_bindings import (
            standalone_request, validate_phase_plan,
        )
        from triton.flagmega.rules.ntt.r004_native_model import native_model_request
        model_contract = native_model_request(module)
        if model_contract is not None:
            if module_kind != "triton" or kernel_parameters.get("variant") != model_contract["implementation"].rsplit(".",1)[-1]:
                raise IRVerificationError("Native model GDN only supports the qualified Triton column path.")
            kernel_parameters["native_model_contract"] = model_contract
        elif module_kind != "triton" or standalone_request(module) is None:
            raise IRVerificationError("r004 reference is standalone Triton only.")
        plan = validate_phase_plan(node.attrs["semantic_attrs"].get("phase_plan"))
        if node.attrs.get("candidate") != plan["implementation"]:
            raise IRVerificationError("r004 phase plan/provider identity mismatch.")
        kernel_parameters["phase_plan"] = plan
        # One backing arena keeps exact signed offsets. Its ten typed F32
        # views/spans/lifetimes live in phase_plan and native definitions.
        expected_workspaces = (WorkspaceRequirement(
            "phase_pool", tensor_type("uint8", (plan["pool"]["nbytes"],)),
            memory_space="workspace", alignment=256),)
        # The generic GDN target attachment predates the phase provider and
        # requests core_scratch for every variant. Replace only that known
        # requirement under this explicit versioned standalone contract.
        legacy_scratch = (WorkspaceRequirement(
            "core_scratch", tensor_type("float32", (48, 128)), alignment=128),)
        if workspaces and workspaces not in (expected_workspaces, legacy_scratch):
            raise IRVerificationError("r004 typed phase workspaces differ from v4.")
        workspaces = expected_workspaces
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
            fused_input_types=(tuple(module.node_map[value].type for value in node.inputs)
                               if node.attrs["semantic_attrs"].get("pre_ops") or node.attrs["semantic_attrs"].get("post_ops")
                               else None),
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
    if module_kind != "triton":
        payload["module_kind"] = module_kind
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
        PrimParameter(name, value_type, PrimParameterRole.OUTPUT, alignment_bytes=_q8_formal_alignment(node, name, result=True))
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
        module_kind=module_kind,
        parameters=(*parameters, *result_parameters, *workspace_parameters),
        dispatch=dispatch,
        results=Return(tuple(
            ReturnBinding(ValueRef(name, value_type), name)
            for name, value_type in result_fields
        )),
        attrs={},
    )
    return function, identity


def r004_phase_definitions(coarse: KernelDefinition) -> tuple[KernelDefinition, ...]:
    """Materialize exact native contracts from a selected coarse phase plan.

    No graph partitioning or synthetic generic kernel ops: each definition
    records its own pool/root effects and the frozen native positional ABI.
    """
    from triton.flagmega.rules.ntt.r004_gdn_phase_bindings import (
        phase_writer_proof, validate_phase_plan,
    )
    plan = validate_phase_plan(coarse.dispatch.parameters.get("phase_plan"))
    if coarse.dispatch.candidate != plan["implementation"]:
        raise IRVerificationError("r004 native definition has the wrong provider.")
    from triton.flagmega.codegen.triton.kernels.gdn_recurrent.r004_contract_v4 import native_arguments
    NATIVE_ARGUMENTS = native_arguments(plan)
    views = {v["byte_offset"]: v for v in plan["pool"]["views"]}
    definitions = []
    for entry in plan["entries"]:
        phase = entry["phase"]
        model_contract = coarse.dispatch.parameters.get("native_model_contract")
        if model_contract is not None:
            from triton.flagmega.rules.ntt.r004_native_model import model_phase_symbol
            symbol = model_phase_symbol(phase, model_contract)
            dependencies = [model_phase_symbol(plan['entries'][i]['phase'], model_contract)
                            for i in range(len(plan['entries']))
                            if plan['entries'][i]['symbol'] in entry['depends_on']]
        else:
            symbol, dependencies = entry['symbol'], entry['depends_on']
        parameters, reads, writes, effects = [], [], [], []
        for i, binding in enumerate(entry["argument_bindings"]):
            descriptor = (views[binding["byte_offset"]] if binding["kind"] == "pointer"
                          else plan["roots"][binding["source"]])
            if model_contract is not None and binding['source'] == 3:
                descriptor = {**descriptor, 'shape':model_contract['native_state_shape'],
                              'nbytes':model_contract['native_state_nbytes']}
            name = NATIVE_ARGUMENTS[phase][i]
            is_write = (descriptor.get("writer") == phase
                        or (phase in ("core", "clear") and binding["source"] == 3)
                        or (phase == "y" and binding["source"] == 5))
            is_read = not is_write or (binding['source']==3 and (phase=='clear' or plan['module'] in ('r004_T_full_p5_v4', 'r004_T_full_p5_v6', 'r017_T_full_p5_w1', 'r017_T_full_p5_w2', 'r017_T_full_p5_w4', 'r017_T_full_p5_w8', 'r017_T_full_p5_w16', 'r017_T_full_p5_w32', 'r017_T_full_p5_w64', 'r017_T_full_p5_split', 'r017_T_full_p5_split4', 'r017_T_full_p5_split8', 'r017_T_full_p5_split4lut') and phase=='core'))
            # ParameterRole describes the canonical logical TIR ABI;
            # memory_effects separately describe physical reads/writes.
            # The explicit native_parameter_order below restores the frozen
            # native positions without weakening PrimFunction ordering.
            role = PrimParameterRole.OUTPUT if is_write else PrimParameterRole.INPUT
            parameters.append(PrimParameter(name, tensor_type(descriptor["dtype"], descriptor["shape"]), role))
            access = (MemoryEffect.READ_WRITE if is_read and is_write
                      else MemoryEffect.WRITE if is_write else MemoryEffect.READ)
            effects.append((name, access))
            if is_read: reads.append(name)
            if is_write: writes.append(name)
        native_order = tuple(p.name for p in parameters)
        parameters = [p for p in parameters if p.role == PrimParameterRole.INPUT] + [
            p for p in parameters if p.role == PrimParameterRole.OUTPUT]
        # A coarse distributed signature has nine semantic inputs. It cannot
        # describe a native phase's three/two/eight pointer positions; phase
        # placement belongs to its own execution mesh and scalar writer proof.
        selected = {**{k: v for k, v in coarse.dispatch.parameters.items() if k != "distribution"}, "r004_phase": phase,
                    "phase_plan": plan, "program_grid": entry["grid"],
                    "program_owner_mapping": "x-fast"}
        dispatch = KernelDispatch(
            semantic_op=coarse.dispatch.semantic_op,
            candidate=plan["implementation"], parameters=selected,
            facts={"internal_grid_barriers": 0, "scalar_writer_proof": phase_writer_proof(phase, plan)},
            arguments=tuple(p.name for p in parameters if p.role != PrimParameterRole.OUTPUT),
            outputs=tuple(p.name for p in parameters if p.role == PrimParameterRole.OUTPUT),
            semantic_attrs=coarse.dispatch.semantic_attrs,
            reads=tuple(reads), writes=tuple(writes), memory_effects=tuple(effects),
            effect_kind=coarse.dispatch.effect_kind if phase in ("core", "clear") else "pure",
            effect_resource=coarse.dispatch.effect_resource if phase in ("core", "clear") else None,
        )
        definitions.append(KernelDefinition(
            name=symbol, module_kind="triton", parameters=tuple(parameters), dispatch=dispatch,
            results=Return(tuple(ReturnBinding(ValueRef(p.name, p.type), p.name)
                                 for p in parameters if p.role == PrimParameterRole.OUTPUT)),
            attrs={"execution_mesh": entry["execution_mesh"], "kernel_program_kind": "ordinary",
                   "kernel_num_warps": entry["num_warps"], "kernel_maxnreg": 80,
                   "kernel_enable_static_shared": entry["enable_static_shared"],
                   "native_parameter_order": list(native_order),
                   "argument_bindings": entry["argument_bindings"],
                   "dynamic_argument_indices": entry["dynamic_argument_indices"],
                   "depends_on": dependencies, "phase_pool": plan["pool"],
                   **({"native_model_contract": model_contract} if model_contract is not None else {})},
        ))
    return tuple(definitions)


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
    if function.module_kind != "triton":
        payload["module_kind"] = function.module_kind
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


def _implementation_memory_effect(
    effect: MemoryEffect, node: Node, module: IRModule, *, is_result: bool = False,
    parameter_name: str | None = None, input_names: tuple[str, ...] | None = None,
) -> MemoryEffect:
    """Expose implementation participation before storage and hazard planning."""

    facts = node.attrs.get("facts", node.attrs.get("semantic_facts", {}))
    return implementation_memory_effect(
        effect, facts, is_result=is_result, stage=module.stage, node_id=node.id,
        parameter_name=parameter_name, input_names=input_names,
    )


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
    fused_input_types: tuple | None = None,
) -> tuple[InplaceAliasCandidate, ...]:
    """Freeze definition-level ParameterInfo aliases into the named TIR ABI."""

    candidates: list[InplaceAliasCandidate] = []

    def add(output_name: str, parameter: ParameterInfo) -> None:
        for index, (actual, name) in enumerate(zip(parameter_infos, parameter_names)):
            if actual is parameter:
                if fused_input_types is not None:
                    output_type = dict(_result_fields(result_type))[output_name]
                    if fused_input_types[index] != output_type:
                        # Base-op aliases prove equal-coordinate read-before-
                        # write, not byte overlap after a boundary repacking.
                        continue
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
