# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit selected backend-kernel operation inside a PrimFunction."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.memory_effect import MemoryAccessMode, MemoryEffect
from triton.flagmega.ir.tir.base import TIRCost, TIRStmt, tir_node
from triton.flagmega.ir.tir.buffer import Buffer
from triton.flagmega.ir.tir.inplace_alias_candidate import InplaceAliasCandidate
from triton.flagmega.ir.tir.microkernel_selection import TIRMicroKernelSelection
from triton.flagmega.ir.tir.workspace_requirement import WorkspaceRequirement


@tir_node("kernel_dispatch")
@dataclass(frozen=True, init=False)
class KernelDispatch(TIRStmt):
    """One semantic TIR kernel with an optional selected implementation.

    ``arguments`` and ``outputs`` name PrimFunction ABI parameters.  Selection
    data lives here exactly once; graph ``tir.call`` nodes carry only a callee.
    Semantic selection and target microkernel selection are deliberately
    separate so canonicalization passes can run between them like nncase.
    """

    semantic_op: str
    semantic_candidate: str
    arguments: tuple[str, ...]
    outputs: tuple[str, ...]
    workspaces: tuple[WorkspaceRequirement, ...] = ()
    inplace_alias_candidates: tuple[InplaceAliasCandidate, ...] | None = None
    semantic_parameters: Mapping[str, object] = field(default_factory=dict)
    semantic_facts: Mapping[str, object] = field(default_factory=dict)
    semantic_attrs: Mapping[str, object] = field(default_factory=dict)
    microkernel: TIRMicroKernelSelection | None = None
    shared_workspace_buffers: tuple[Buffer, ...] = ()
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    memory_effects: tuple[tuple[str, MemoryEffect], ...] = ()
    effect_kind: str = "pure"
    effect_resource: str | None = None

    def __init__(
        self,
        semantic_op: str,
        semantic_candidate: str | None = None,
        arguments: tuple[str, ...] = (),
        outputs: tuple[str, ...] = (),
        workspaces: tuple[WorkspaceRequirement, ...] = (),
        inplace_alias_candidates: tuple[InplaceAliasCandidate, ...] | None = None,
        semantic_parameters: Mapping[str, object] | None = None,
        semantic_facts: Mapping[str, object] | None = None,
        semantic_attrs: Mapping[str, object] | None = None,
        microkernel: TIRMicroKernelSelection | None = None,
        shared_workspace_buffers: tuple[Buffer, ...] = (),
        reads: tuple[str, ...] = (),
        writes: tuple[str, ...] = (),
        memory_effects: tuple[tuple[str, MemoryEffect], ...] = (),
        effect_kind: str = "pure",
        effect_resource: str | None = None,
        *,
        candidate: str | None = None,
        parameters: Mapping[str, object] | None = None,
        facts: Mapping[str, object] | None = None,
    ) -> None:
        """Build the split representation, accepting old Python dumps.

        ``candidate/parameters/facts`` is the pre-split checkpoint spelling.
        It is upgraded to a concrete ``TIRMicroKernelSelection`` at load time;
        new emitters serialize only the split fields.
        """

        if candidate is not None:
            if microkernel is not None:
                raise IRSchemaError(
                    "KernelDispatch cannot specify both legacy candidate and microkernel."
                )
            legacy_parameters = dict(parameters or {})
            legacy_facts = dict(facts or {})
            requirements = legacy_facts.pop("requires", ())
            microkernel = TIRMicroKernelSelection(
                implementation=str(candidate),
                family=str(legacy_parameters.get("family", semantic_op)),
                variant=str(legacy_parameters.get("variant", candidate)),
                parameters=legacy_parameters,
                facts=legacy_facts,
                requires=requirements,
            )
            semantic_candidate = semantic_candidate or str(candidate)
        elif parameters is not None or facts is not None:
            if microkernel is None:
                raise IRSchemaError(
                    "Legacy KernelDispatch parameters/facts require candidate."
                )
            # ``dataclasses.replace(dispatch, parameters=...)`` was part of the
            # editable-checkpoint API before semantic and physical selection
            # were split.  Preserve that source compatibility while ensuring
            # the override can only mutate the physical selection.
            physical_parameters = _physical_override(
                parameters,
                semantic_parameters or {},
                current=microkernel.parameters,
                owner="KernelDispatch parameters",
            )
            physical_facts = _physical_override(
                facts,
                semantic_facts or {},
                current=microkernel.facts,
                owner="KernelDispatch facts",
            )
            requirements = physical_facts.pop("requires", microkernel.requires)
            microkernel = TIRMicroKernelSelection(
                implementation=microkernel.implementation,
                family=microkernel.family,
                variant=microkernel.variant,
                parameters=physical_parameters,
                facts=physical_facts,
                requires=requirements,
            )
        object.__setattr__(self, "semantic_op", semantic_op)
        object.__setattr__(
            self,
            "semantic_candidate",
            semantic_op if semantic_candidate is None else semantic_candidate,
        )
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "workspaces", workspaces)
        object.__setattr__(self, "inplace_alias_candidates", inplace_alias_candidates)
        object.__setattr__(self, "semantic_parameters", semantic_parameters or {})
        object.__setattr__(self, "semantic_facts", semantic_facts or {})
        object.__setattr__(self, "semantic_attrs", semantic_attrs or {})
        object.__setattr__(self, "microkernel", microkernel)
        object.__setattr__(self, "shared_workspace_buffers", shared_workspace_buffers)
        object.__setattr__(self, "reads", reads)
        object.__setattr__(self, "writes", writes)
        object.__setattr__(self, "memory_effects", memory_effects)
        object.__setattr__(self, "effect_kind", effect_kind)
        object.__setattr__(self, "effect_resource", effect_resource)
        self.__post_init__()

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", tuple(self.arguments))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "workspaces", tuple(self.workspaces))
        if self.inplace_alias_candidates is not None:
            object.__setattr__(
                self,
                "inplace_alias_candidates",
                tuple(self.inplace_alias_candidates),
            )
        object.__setattr__(
            self, "shared_workspace_buffers", tuple(self.shared_workspace_buffers)
        )
        object.__setattr__(self, "reads", tuple(self.reads))
        object.__setattr__(self, "writes", tuple(self.writes))
        object.__setattr__(self, "memory_effects", tuple(
            (str(name), effect if isinstance(effect, MemoryEffect) else MemoryEffect.from_data(effect))
            for name, effect in self.memory_effects
        ))
        object.__setattr__(
            self, "semantic_parameters", _freeze_mapping(self.semantic_parameters)
        )
        object.__setattr__(self, "semantic_facts", _freeze_mapping(self.semantic_facts))
        object.__setattr__(self, "semantic_attrs", _freeze_mapping(self.semantic_attrs))
        if not self.semantic_op or not self.semantic_candidate:
            raise IRSchemaError(
                "KernelDispatch requires semantic_op and semantic_candidate."
            )
        if not self.outputs or any(not value for value in (*self.arguments, *self.outputs)):
            raise IRSchemaError("KernelDispatch requires named arguments and at least one output.")
        if len({value.name for value in self.workspaces}) != len(self.workspaces):
            raise IRSchemaError("KernelDispatch workspace names must be unique.")
        if self.inplace_alias_candidates is not None:
            if any(
                not isinstance(value, InplaceAliasCandidate)
                for value in self.inplace_alias_candidates
            ):
                raise IRSchemaError(
                    "KernelDispatch inplace aliases must be typed InplaceAliasCandidates."
                )
            pairs = tuple(
                (value.output, value.input)
                for value in self.inplace_alias_candidates
            )
            if len(set(pairs)) != len(pairs):
                raise IRSchemaError(
                    "KernelDispatch inplace alias candidates must be unique."
                )
            unknown_outputs = sorted(
                {value.output for value in self.inplace_alias_candidates}
                - set(self.outputs)
            )
            unknown_inputs = sorted(
                {value.input for value in self.inplace_alias_candidates}
                - set(self.arguments)
            )
            if unknown_outputs or unknown_inputs:
                raise IRSchemaError(
                    "KernelDispatch inplace alias candidates reference values outside "
                    f"its ABI: outputs={unknown_outputs}, inputs={unknown_inputs}."
                )
        if any(
            not isinstance(value, Buffer) for value in self.shared_workspace_buffers
        ):
            raise IRSchemaError(
                "KernelDispatch shared workspace buffers must be typed TIR Buffers."
            )
        shared_names = tuple(value.name for value in self.shared_workspace_buffers)
        if len(set(shared_names)) != len(shared_names):
            raise IRSchemaError(
                "KernelDispatch shared workspace buffer names must be unique."
            )
        if self.microkernel is None and self.shared_workspace_buffers:
            raise IRSchemaError(
                "KernelDispatch cannot materialize shared workspaces without a microkernel."
            )
        if self.microkernel is not None and self.shared_workspace_buffers:
            descriptors = self.microkernel.shared_workspaces
            if len(descriptors) != len(self.shared_workspace_buffers) or any(
                descriptor.name != buffer.name
                or descriptor.type != buffer.type
                or buffer.mem_span.buffer.memory_space != "shared"
                or buffer.mem_span.buffer.alignment < descriptor.alignment_bytes
                for descriptor, buffer in zip(
                    descriptors, self.shared_workspace_buffers
                )
            ):
                raise IRSchemaError(
                    "KernelDispatch shared workspace buffers differ from the selected "
                    "microkernel descriptors."
                )
        available = set(self.arguments) | set(self.outputs)
        if not set(self.reads).issubset(available) or not set(self.writes).issubset(available):
            raise IRSchemaError("KernelDispatch effects must reference ABI argument/output names.")
        if not self.effect_kind:
            raise IRSchemaError("KernelDispatch requires a non-empty effect kind.")
        effect_names = tuple(name for name, _ in self.memory_effects)
        if len(set(effect_names)) != len(effect_names) or not set(effect_names).issubset(available):
            raise IRSchemaError(
                "KernelDispatch memory effects require unique ABI argument/output names."
            )
        if self.memory_effects:
            effect_map = dict(self.memory_effects)
            effect_reads = {
                name for name, value in effect_map.items()
                if value.physical_mode & MemoryAccessMode.READ
            }
            effect_writes = {
                name for name, value in effect_map.items()
                if value.physical_mode & MemoryAccessMode.WRITE
            }
            if effect_reads != set(self.reads) or effect_writes != set(self.writes):
                raise IRSchemaError(
                    "KernelDispatch reads/writes disagree with its typed memory effects."
                )

    @property
    def memory_effect_map(self) -> Mapping[str, MemoryEffect]:
        """Typed effects keyed by PrimFunction ABI name.

        Legacy checkpoints did not carry rich effects.  Their existing
        reads/writes remain a conservative editable fallback.
        """

        if self.memory_effects:
            return MappingProxyType(dict(self.memory_effects))
        return MappingProxyType({
            name: MemoryEffect(
                (MemoryAccessMode.READ if name in self.reads else MemoryAccessMode.NONE)
                | (MemoryAccessMode.WRITE if name in self.writes else MemoryAccessMode.NONE)
            )
            for name in dict.fromkeys((*self.arguments, *self.outputs))
            if name in self.reads or name in self.writes
        })

    @property
    def local_cost(self) -> TIRCost:
        return TIRCost(
            flops=_optional_factor(self.resolved_facts, "flops"),
            bytes_read=_optional_factor(self.resolved_facts, "bytes_read"),
            bytes_written=_optional_factor(self.resolved_facts, "bytes_written"),
            synchronizations=_optional_factor(self.resolved_facts, "synchronizations"),
        )

    @property
    def candidate(self) -> str:
        """Compatibility view used by existing codegen consumers."""

        return (
            self.semantic_candidate
            if self.microkernel is None
            else self.microkernel.implementation
        )

    @property
    def parameters(self) -> Mapping[str, object]:
        return self.resolved_parameters

    @property
    def facts(self) -> Mapping[str, object]:
        return self.resolved_facts

    @property
    def resolved_parameters(self) -> Mapping[str, object]:
        physical = {}
        if self.microkernel is not None:
            physical = dict(self.microkernel.parameters)
            identities = {
                "family": self.microkernel.family,
                "variant": self.microkernel.variant,
            }
            conflicts = {
                key: (physical[key], value)
                for key, value in identities.items()
                if key in physical and physical[key] != value
            }
            if conflicts:
                raise IRSchemaError(
                    "KernelDispatch microkernel parameters conflict with its "
                    f"typed identity: {conflicts}."
                )
            physical = {**identities, **physical}
        return _merge_disjoint(
            physical,
            self.semantic_parameters,
            owner="KernelDispatch parameters",
        )

    @property
    def resolved_facts(self) -> Mapping[str, object]:
        physical = {} if self.microkernel is None else dict(self.microkernel.facts)
        result = _merge_disjoint(
            physical,
            self.semantic_facts,
            owner="KernelDispatch facts",
        )
        if self.microkernel is not None and self.microkernel.requires:
            result = _merge_disjoint(
                dict(result),
                {"requires": self.microkernel.requires},
                owner="KernelDispatch requirements",
            )
        return result


def kernel_dispatch_of(function) -> KernelDispatch | None:
    """Return the canonical selected-kernel body, or ``None`` for general TIR."""

    from triton.flagmega.ir.tir.sequential import Sequential
    from triton.flagmega.ir.tir.kernel_definition import KernelDefinition

    if isinstance(function, KernelDefinition):
        return function.dispatch

    if not isinstance(function.body, Sequential) or len(function.body.fields) != 1:
        return None
    statement = function.body.fields[0]
    if isinstance(statement, KernelDispatch):
        return statement

    # Post-bufferize transfer-pipeline lowering keeps the same canonical
    # selected-kernel identity under explicit producer/consumer role views.
    # Multi-stage regions are general TIR functions and intentionally do not
    # masquerade as the one-kernel form consumed by graph-call codegen.
    from triton.flagmega.ir.tir.pipeline_stage import PipelineStage
    from triton.flagmega.ir.tir.producer_consumer_region import (
        ProducerConsumerRegion,
    )

    if not isinstance(statement, ProducerConsumerRegion):
        return None
    stages = tuple(
        field
        for field in statement.consume_body.fields
        if isinstance(field, PipelineStage)
    )
    if len(stages) != 1:
        return None
    operation = stages[0].operation
    return operation if isinstance(operation, KernelDispatch) else None


def kernel_dispatch_for_call(module, node) -> KernelDispatch | None:
    """Resolve a graph ``tir.call`` to its selected kernel operation."""

    if node.op != "tir.call":
        return None
    function = module.kernel_callable_map.get(str(node.attrs.get("callee", "")))
    return None if function is None else kernel_dispatch_of(function)


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({str(key): _freeze(item) for key, item in sorted(value.items())})


def _optional_factor(facts: Mapping[str, object], name: str) -> int | None:
    return None if name not in facts or facts[name] is None else int(facts[name])


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _merge_disjoint(
    lhs: Mapping[str, object],
    rhs: Mapping[str, object],
    *,
    owner: str,
) -> Mapping[str, object]:
    conflicts = {
        key: (lhs[key], value)
        for key, value in rhs.items()
        if key in lhs and lhs[key] != value
    }
    if conflicts:
        raise IRSchemaError(f"{owner} contain conflicting values: {conflicts}.")
    return _freeze_mapping({**dict(lhs), **dict(rhs)})


def _physical_override(
    override: Mapping[str, object] | None,
    semantic: Mapping[str, object],
    *,
    current: Mapping[str, object],
    owner: str,
) -> dict[str, object]:
    if override is None:
        return dict(current)
    result = dict(override)
    conflicts = {
        key: (result[key], value)
        for key, value in semantic.items()
        if key in result and result[key] != value
    }
    if conflicts:
        raise IRSchemaError(f"{owner} cannot override semantic values: {conflicts}.")
    for key in semantic:
        result.pop(key, None)
    return result


__all__ = ["KernelDispatch", "kernel_dispatch_for_call", "kernel_dispatch_of"]
