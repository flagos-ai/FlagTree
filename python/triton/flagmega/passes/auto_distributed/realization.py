# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Target policy for physically realizing a semantic distributed reshard."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from triton.flagmega.ir import (
    DistributedType,
    IRType,
    SBPSplit,
    TensorType,
    is_local_shard_subview,
    sharded_view_error,
)


class DistributedReshardRealization(str, Enum):
    BOXING = "boxing"
    SHARDED_VIEW = "sharded_view"
    UNSUPPORTED = "unsupported"


class DistributedReshardSourceKind(str, Enum):
    CONSTANT = "constant"
    INTERNAL = "internal"
    FUNCTION_PARAMETER = "function_parameter"


class DistributedReshardUsageKind(str, Enum):
    INTERNAL = "internal"
    FUNCTION_BOUNDARY = "function_boundary"
    PROGRAM_OUTPUT = "program_output"


@dataclass(frozen=True)
class DistributedReshardRealizationContext:
    source_type: IRType
    target_type: IRType
    source_kind: DistributedReshardSourceKind
    usage_kind: DistributedReshardUsageKind


class DistributedReshardRealizationPolicy(Protocol):
    def uses_sharded_views_for_constants(self) -> bool: ...

    def classify(
        self,
        context: DistributedReshardRealizationContext,
    ) -> DistributedReshardRealization: ...


class NttDistributedReshardRealizationPolicy:
    """Reshard realization shared by NTT targets with unified constant storage."""

    def __init__(self, *, unified_shared_storage: bool = False) -> None:
        self._unified_shared_storage = bool(unified_shared_storage)

    def uses_sharded_views_for_constants(self) -> bool:
        return self._unified_shared_storage

    def classify(
        self,
        context: DistributedReshardRealizationContext,
    ) -> DistributedReshardRealization:
        from triton.flagmega.passes.auto_distributed.reshard import can_box
        from triton.flagmega.ir import TensorType

        if not can_box(context.source_type, context.target_type):
            return DistributedReshardRealization.UNSUPPORTED
        if self.can_alias_constant(context):
            return DistributedReshardRealization.SHARDED_VIEW
        return DistributedReshardRealization.BOXING

    def can_alias_constant(
        self,
        context: DistributedReshardRealizationContext,
    ) -> bool:
        if (
            not self._unified_shared_storage
            or context.source_kind != DistributedReshardSourceKind.CONSTANT
            or not isinstance(context.target_type, DistributedType)
            or sharded_view_error(context.source_type, context.target_type) is not None
        ):
            return False
        if isinstance(context.source_type, TensorType):
            return True
        return (
            isinstance(context.source_type, DistributedType)
            and is_local_shard_subview(context.source_type, context.target_type)
        )


class PyNttDistributedReshardRealizationPolicy(NttDistributedReshardRealizationPolicy):
    """PyNTT alias policy for its persistent unified-memory execution model."""

    def __init__(self) -> None:
        super().__init__(unified_shared_storage=True)

    def classify(
        self,
        context: DistributedReshardRealizationContext,
    ) -> DistributedReshardRealization:
        from triton.flagmega.passes.auto_distributed.reshard import can_box

        if not can_box(context.source_type, context.target_type):
            return DistributedReshardRealization.UNSUPPORTED
        if (
            not isinstance(context.target_type, DistributedType)
            or sharded_view_error(context.source_type, context.target_type) is not None
        ):
            return DistributedReshardRealization.BOXING
        if self.can_alias_constant(context):
            return DistributedReshardRealization.SHARDED_VIEW
        if (
            context.usage_kind not in {
                DistributedReshardUsageKind.INTERNAL,
                DistributedReshardUsageKind.PROGRAM_OUTPUT,
            }
            or not isinstance(context.source_type, DistributedType)
        ):
            return DistributedReshardRealization.BOXING

        source_type = context.source_type
        target_type = context.target_type
        if (
            is_local_shard_subview(source_type, target_type)
            and _preserves_non_block_owners(source_type, target_type)
        ):
            return DistributedReshardRealization.SHARDED_VIEW
        if context.source_kind != DistributedReshardSourceKind.INTERNAL:
            return DistributedReshardRealization.BOXING
        return (
            DistributedReshardRealization.SHARDED_VIEW
            if _can_materialize_canonical_chip_view(source_type, target_type)
            else DistributedReshardRealization.BOXING
        )


def _placement_axis_owners(value: DistributedType) -> tuple[int, ...] | None:
    owners = [-1] * value.placement.rank
    for tensor_axis, policy in enumerate(value.axis_policies):
        if not isinstance(policy, SBPSplit):
            continue
        for placement_axis in policy.hierarchy_axes:
            if placement_axis >= len(owners) or owners[placement_axis] >= 0:
                return None
            owners[placement_axis] = tensor_axis
    return tuple(owners)


def _preserves_non_block_owners(
    source_type: DistributedType,
    target_type: DistributedType,
) -> bool:
    source_owners = _placement_axis_owners(source_type)
    target_owners = _placement_axis_owners(target_type)
    if source_owners is None or target_owners is None:
        return False
    return all(
        source_type.placement.hierarchy[axis] <= 1
        or source_type.placement.is_physical_block_axis(axis)
        or source_owners[axis] == target_owners[axis]
        for axis in range(source_type.placement.rank)
    )


def _can_materialize_canonical_chip_view(
    source_type: DistributedType,
    target_type: DistributedType,
) -> bool:
    return _preserves_non_block_owners(source_type, target_type)


class CanonicalStorageReshardRealizationPolicy(PyNttDistributedReshardRealizationPolicy):
    """Compatibility spelling for the former PyNTT-only policy name."""


__all__ = [
    "CanonicalStorageReshardRealizationPolicy",
    "DistributedReshardRealization",
    "DistributedReshardRealizationContext",
    "DistributedReshardRealizationPolicy",
    "DistributedReshardSourceKind",
    "DistributedReshardUsageKind",
    "NttDistributedReshardRealizationPolicy",
    "PyNttDistributedReshardRealizationPolicy",
]
