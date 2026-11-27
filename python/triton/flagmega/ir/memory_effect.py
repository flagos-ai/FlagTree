# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Operand memory-effect contracts shared by op definitions and TIR.

The representation follows nncase's separation of access mode, visibility
scope, reduction lifetime, owner domain and logical subresource.  These are IR
facts rather than codegen hints: they survive Python dump/edit/resume and are
copied into a materialized :class:`KernelDispatch`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum, IntFlag
from typing import ClassVar

from triton.flagmega.errors import IRSchemaError, IRVerificationError


class MemoryAccessMode(IntFlag):
    NONE = 0
    READ = 1
    WRITE = 2
    READ_WRITE = READ | WRITE


class MemoryAccessScope(str, Enum):
    INFERRED = "inferred"
    BLOCK = "block"
    CHIP = "chip"
    DEVICE = "device"


class MemoryEffectKind(str, Enum):
    DIRECT = "direct"
    REDUCTION_ACCUMULATOR = "reduction_accumulator"


class MemoryAccessDomainKind(str, Enum):
    ALL_BLOCKS = "all_blocks"
    FIXED_BLOCK = "fixed_block"


@dataclass(frozen=True)
class MemoryAccessDomain:
    kind: MemoryAccessDomainKind = MemoryAccessDomainKind.ALL_BLOCKS
    block_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", MemoryAccessDomainKind(self.kind))
        if self.kind is MemoryAccessDomainKind.ALL_BLOCKS:
            if self.block_index is not None:
                raise IRSchemaError("All-block memory domain cannot name a block.")
        elif (
            isinstance(self.block_index, bool)
            or not isinstance(self.block_index, int)
            or self.block_index < 0
        ):
            raise IRSchemaError("Fixed-block memory domain requires a non-negative index.")

    @classmethod
    def fixed_block(cls, block_index: int) -> MemoryAccessDomain:
        return cls(MemoryAccessDomainKind.FIXED_BLOCK, block_index)

    def is_same_fixed_block(self, other: MemoryAccessDomain) -> bool:
        return self.kind is MemoryAccessDomainKind.FIXED_BLOCK and self == other

    def to_data(self) -> dict[str, object]:
        return {"kind": self.kind.value, "block_index": self.block_index}

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> MemoryAccessDomain:
        raw_index = data.get("block_index")
        return cls(
            MemoryAccessDomainKind(str(data.get("kind", "all_blocks"))),
            None if raw_index is None else int(raw_index),
        )


class MemoryAccessPartitionKind(str, Enum):
    WHOLE_RESOURCE = "whole_resource"
    ARGUMENT = "argument"


@dataclass(frozen=True)
class MemoryAccessPartition:
    kind: MemoryAccessPartitionKind = MemoryAccessPartitionKind.WHOLE_RESOURCE
    argument_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", MemoryAccessPartitionKind(self.kind))
        if self.kind is MemoryAccessPartitionKind.WHOLE_RESOURCE:
            if self.argument_index is not None:
                raise IRSchemaError("Whole-resource partition cannot name an argument.")
        elif (
            isinstance(self.argument_index, bool)
            or not isinstance(self.argument_index, int)
            or self.argument_index < 0
        ):
            raise IRSchemaError("Argument partition requires a non-negative operand index.")

    @classmethod
    def by_argument(cls, argument_index: int) -> MemoryAccessPartition:
        return cls(MemoryAccessPartitionKind.ARGUMENT, argument_index)

    def to_data(self) -> dict[str, object]:
        return {"kind": self.kind.value, "argument_index": self.argument_index}

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> MemoryAccessPartition:
        raw_index = data.get("argument_index")
        return cls(
            MemoryAccessPartitionKind(str(data.get("kind", "whole_resource"))),
            None if raw_index is None else int(raw_index),
        )


class MemoryOwnerAccess(str, Enum):
    LOCAL = "local"
    PARTIAL_GROUP = "partial_group"


@dataclass(frozen=True)
class MemoryEffect:
    """Possible physical accesses through one operation operand."""

    mode: MemoryAccessMode = MemoryAccessMode.NONE
    scope: MemoryAccessScope = MemoryAccessScope.INFERRED
    kind: MemoryEffectKind = MemoryEffectKind.DIRECT
    access_domain: MemoryAccessDomain = MemoryAccessDomain()
    access_partition: MemoryAccessPartition = MemoryAccessPartition()
    owner_access: MemoryOwnerAccess = MemoryOwnerAccess.LOCAL
    field_effects: tuple[tuple[str, MemoryEffect], ...] = ()

    NONE: ClassVar[MemoryEffect]
    READ: ClassVar[MemoryEffect]
    WRITE: ClassVar[MemoryEffect]
    READ_WRITE: ClassVar[MemoryEffect]
    CHIP_READ: ClassVar[MemoryEffect]
    CHIP_WRITE: ClassVar[MemoryEffect]
    CHIP_READ_WRITE: ClassVar[MemoryEffect]
    REDUCTION_WRITE: ClassVar[MemoryEffect]
    REDUCTION_READ_WRITE: ClassVar[MemoryEffect]

    def __post_init__(self) -> None:
        mode = self.mode
        if isinstance(mode, str):
            try:
                mode = MemoryAccessMode[mode.upper()]
            except KeyError as error:
                raise IRSchemaError(f"Unknown memory access mode {mode!r}.") from error
        object.__setattr__(self, "mode", MemoryAccessMode(mode))
        object.__setattr__(self, "scope", MemoryAccessScope(self.scope))
        object.__setattr__(self, "kind", MemoryEffectKind(self.kind))
        object.__setattr__(self, "access_domain", _domain(self.access_domain))
        object.__setattr__(self, "access_partition", _partition(self.access_partition))
        object.__setattr__(self, "owner_access", MemoryOwnerAccess(self.owner_access))
        fields = tuple(self.field_effects)
        if any(not isinstance(name, str) or not name or not isinstance(value, MemoryEffect) for name, value in fields):
            raise IRSchemaError("Reference field effects require non-empty names and typed effects.")
        if len({name for name, _ in fields}) != len(fields):
            raise IRSchemaError("Reference field effects require unique names.")
        if fields:
            aggregate = MemoryAccessMode.NONE
            for _, value in fields:
                aggregate |= value.mode
            if self.mode != aggregate:
                raise IRSchemaError("Reference field effects disagree with their aggregate mode.")
            if (self.scope is not MemoryAccessScope.INFERRED or self.kind is not MemoryEffectKind.DIRECT
                    or self.access_domain != MemoryAccessDomain() or self.access_partition != MemoryAccessPartition()
                    or self.owner_access is not MemoryOwnerAccess.LOCAL):
                raise IRSchemaError("Reference field refinements must be declared on the field effects.")
        object.__setattr__(self, "field_effects", tuple(sorted(fields)))

    @classmethod
    def for_fields(cls, **effects: MemoryEffect) -> MemoryEffect:
        """Specify Ref fields independently; omitted fields have no access."""
        mode = MemoryAccessMode.NONE
        for value in effects.values():
            if not isinstance(value, cls):
                raise IRSchemaError("Reference field effects must be typed MemoryEffects.")
            mode |= value.mode
        return cls(mode, field_effects=tuple(effects.items()))

    @property
    def value(self) -> str:
        """Compatibility spelling of the former mode-only enum."""

        return self.mode.name.lower()

    @property
    def physical_mode(self) -> MemoryAccessMode:
        """Return only accesses which reach the physical buffer.

        Reduction feedback is private to the backend reduction region; only
        its final write is externally observable, matching nncase's
        ``MemoryEffectUtility.GetPhysicalBufferAccessMode``.
        """

        if self.field_effects:
            mode = MemoryAccessMode.NONE
            for _, effect in self.field_effects:
                mode |= effect.physical_mode
            return mode
        if self.kind is MemoryEffectKind.REDUCTION_ACCUMULATOR:
            return self.mode & MemoryAccessMode.WRITE
        return self.mode

    def in_fixed_block(self, block_index: int) -> MemoryEffect:
        if self.field_effects:
            return self.for_fields(**{name: value.in_fixed_block(block_index) for name, value in self.field_effects})
        return MemoryEffect(
            self.mode, self.scope, self.kind,
            MemoryAccessDomain.fixed_block(block_index),
            self.access_partition, self.owner_access,
        )

    def partitioned_by_argument(self, argument_index: int) -> MemoryEffect:
        if self.field_effects:
            return self.for_fields(**{name: value.partitioned_by_argument(argument_index)
                                     for name, value in self.field_effects})
        return MemoryEffect(
            self.mode, self.scope, self.kind, self.access_domain,
            MemoryAccessPartition.by_argument(argument_index), self.owner_access,
        )

    def across_partial_owners(self) -> MemoryEffect:
        if self.field_effects:
            return self.for_fields(**{name: value.across_partial_owners() for name, value in self.field_effects})
        return MemoryEffect(
            self.mode, self.scope, self.kind, self.access_domain,
            self.access_partition, MemoryOwnerAccess.PARTIAL_GROUP,
        )

    def to_data(self) -> dict[str, object]:
        result = {
            "mode": self.mode.name.lower(),
            "scope": self.scope.value,
            "kind": self.kind.value,
            "access_domain": self.access_domain.to_data(),
            "access_partition": self.access_partition.to_data(),
            "owner_access": self.owner_access.value,
        }
        if self.field_effects:
            result["field_effects"] = {name: value.to_data() for name, value in self.field_effects}
        return result

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> MemoryEffect:
        raw_domain = data.get("access_domain", {})
        raw_partition = data.get("access_partition", {})
        if not isinstance(raw_domain, Mapping) or not isinstance(raw_partition, Mapping):
            raise IRSchemaError("MemoryEffect domain and partition must be mappings.")
        fields = data.get("field_effects", {})
        if not isinstance(fields, Mapping) or any(not isinstance(value, Mapping) for value in fields.values()):
            raise IRSchemaError("MemoryEffect field effects must be mappings.")
        return cls(
            str(data.get("mode", "none")),
            MemoryAccessScope(str(data.get("scope", "inferred"))),
            MemoryEffectKind(str(data.get("kind", "direct"))),
            MemoryAccessDomain.from_data(raw_domain),
            MemoryAccessPartition.from_data(raw_partition),
            MemoryOwnerAccess(str(data.get("owner_access", "local"))),
            tuple((name, cls.from_data(value)) for name, value in fields.items()),
        )


def expand_memory_effect(value_type, effect: MemoryEffect) -> tuple[MemoryEffect, ...]:
    """Resolve typed effects in the same leaf order as the buffer ABI."""
    from triton.flagmega.ir.model import DistributedType, NoneType, RefType, TensorType, TupleType

    if isinstance(value_type, RefType):
        fields = dict(effect.field_effects)
        missing = fields.keys() - {name for name, _ in value_type.fields}
        if missing:
            raise IRSchemaError(f"Memory effect names unknown Ref fields: {sorted(missing)}.")
        return tuple(leaf for name, field in value_type.fields
                     for leaf in expand_memory_effect(field, fields.get(name, MemoryEffect.NONE) if fields else effect))
    if effect.field_effects:
        raise IRSchemaError("Field memory effects require a Ref type.")
    if isinstance(value_type, TupleType):
        return tuple(leaf for field in value_type.fields for leaf in expand_memory_effect(field, effect))
    if isinstance(value_type, NoneType):
        return ()
    if isinstance(value_type, (TensorType, DistributedType)):
        return (effect,)
    raise IRSchemaError(f"Cannot expand a memory effect for {type(value_type).__name__}.")


def memory_effect(value: MemoryEffect | MemoryAccessMode | str) -> MemoryEffect:
    if isinstance(value, MemoryEffect):
        return value
    return MemoryEffect(value)


def _domain(value: MemoryAccessDomain | Mapping[str, object]) -> MemoryAccessDomain:
    return value if isinstance(value, MemoryAccessDomain) else MemoryAccessDomain.from_data(value)


def _partition(
    value: MemoryAccessPartition | Mapping[str, object],
) -> MemoryAccessPartition:
    return value if isinstance(value, MemoryAccessPartition) else MemoryAccessPartition.from_data(value)


MemoryEffect.NONE = MemoryEffect(MemoryAccessMode.NONE)
MemoryEffect.READ = MemoryEffect(MemoryAccessMode.READ)
MemoryEffect.WRITE = MemoryEffect(MemoryAccessMode.WRITE)
MemoryEffect.READ_WRITE = MemoryEffect(MemoryAccessMode.READ_WRITE)
MemoryEffect.CHIP_READ = MemoryEffect(MemoryAccessMode.READ, MemoryAccessScope.CHIP)
MemoryEffect.CHIP_WRITE = MemoryEffect(MemoryAccessMode.WRITE, MemoryAccessScope.CHIP)
MemoryEffect.CHIP_READ_WRITE = MemoryEffect(
    MemoryAccessMode.READ_WRITE, MemoryAccessScope.CHIP
)
MemoryEffect.REDUCTION_WRITE = MemoryEffect(
    MemoryAccessMode.WRITE,
    kind=MemoryEffectKind.REDUCTION_ACCUMULATOR,
)
MemoryEffect.REDUCTION_READ_WRITE = MemoryEffect(
    MemoryAccessMode.READ_WRITE,
    kind=MemoryEffectKind.REDUCTION_ACCUMULATOR,
)


def implementation_memory_effect(
    effect: MemoryEffect,
    facts: Mapping[str, object],
    *,
    is_result: bool = False,
    parameter_name: str | None = None,
    input_names: tuple[str, ...] | None = None,
    stage: str | None = None,
    node_id: str | None = None,
) -> MemoryEffect:
    """Refine an op effect with the selected implementation's physical contract.

    ``source_memory_scopes`` overrides the uniform source scope by ABI input
    name. An inferred override retains the op-local requirement; it does not
    erase a scope required by the semantic operation. Unnamed inputs retain
    the uniform declaration. The full input schema rejects misspelled names.
    """

    scopes = _implementation_source_scopes(facts, input_names, stage=stage, node_id=node_id)
    if not is_result and scopes and parameter_name not in input_names:
        raise IRVerificationError("Implementation source_memory_scopes requires the current input name.",
                                  stage=stage, node_id=node_id)

    if effect.physical_mode != MemoryAccessMode.NONE:
        declared = MemoryAccessScope(
            facts.get("result_memory_scope", "inferred") if is_result
            else scopes.get(parameter_name, facts.get("source_memory_scope", "inferred")))
        if declared is not MemoryAccessScope.INFERRED:
            effect = replace(effect, scope=declared)
    scope = facts.get("participant_scope", "all_programs")
    if scope not in {"all_programs", "single_program"}:
        raise IRVerificationError(
            f"Unknown implementation participant_scope {scope!r}.",
            stage=stage, node_id=node_id,
        )
    if scope == "all_programs" or effect.physical_mode == MemoryAccessMode.NONE:
        return effect
    if (
        effect.access_domain.kind is MemoryAccessDomainKind.FIXED_BLOCK
        and effect.access_domain.block_index != 0
    ):
        raise IRVerificationError(
            "Single-program implementation conflicts with the op access domain.",
            stage=stage, node_id=node_id,
        )
    return effect.in_fixed_block(0)


def _implementation_source_scopes(facts, input_names, *, stage, node_id):
    if "source_memory_scopes" not in facts:
        return {}
    raw = facts["source_memory_scopes"]

    def reject(reason):
        raise IRVerificationError(f"Implementation source_memory_scopes {reason}.", stage=stage, node_id=node_id)

    if not isinstance(raw, Mapping) or any(not isinstance(name, str) for name in raw):
        reject("must be a mapping from input names to memory scopes")
    if input_names is None:
        reject("requires the complete input parameter schema")
    unknown = raw.keys() - set(input_names)
    if unknown:
        reject(f"names unknown inputs {sorted(unknown)}")
    try:
        return {name: MemoryAccessScope(scope) for name, scope in raw.items()}
    except (ValueError, TypeError):
        reject("contains an invalid memory scope")


__all__ = [
    "MemoryAccessDomain",
    "MemoryAccessDomainKind",
    "MemoryAccessMode",
    "MemoryAccessPartition",
    "MemoryAccessPartitionKind",
    "MemoryAccessScope",
    "MemoryEffect",
    "MemoryEffectKind",
    "MemoryOwnerAccess",
    "expand_memory_effect",
    "implementation_memory_effect",
    "memory_effect",
]
