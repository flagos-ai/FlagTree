# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Physical costs for materializing semantic distributed reshard edges."""

from __future__ import annotations

from dataclasses import dataclass

from triton.flagmega.ir import DistributedType, Node, is_local_shard_subview
from triton.flagmega.ir.ops.distributed.boxing import Boxing
from triton.flagmega.passes.auto_distributed.operation_cost import DistributedOperationCostModel
from triton.flagmega.passes.auto_distributed.realization import (
    DistributedReshardRealization,
    DistributedReshardRealizationContext,
    DistributedReshardSourceKind,
    DistributedReshardUsageKind,
)
from triton.flagmega.passes.auto_distributed.reshard import reshard_step_cost


@dataclass(frozen=True)
class DistributedReshardCostModel:
    """Map one physical realization to the AutoDistribution objective.

    A view moves no bytes, but an internal view which widens one owner's
    visibility publishes canonical storage to other blocks. That publication
    is one grid synchronization. The target supplies its objective weight, so
    graph rules contain no model or hardware special case.
    """

    grid_synchronization_cost: int = 2200
    operation_cost_model: DistributedOperationCostModel | None = None

    def __post_init__(self) -> None:
        value = self.grid_synchronization_cost
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "grid_synchronization_cost must be a non-negative integer."
            )

    def realization_cost(
        self,
        context: DistributedReshardRealizationContext,
        realization: DistributedReshardRealization,
    ) -> int:
        if realization == DistributedReshardRealization.UNSUPPORTED:
            return 2_000_000_000
        if realization == DistributedReshardRealization.BOXING:
            if self.operation_cost_model is not None:
                source = Node("<collective-source>", "builtin.var", (), context.source_type)
                factors = Boxing.cost_factors((source,), {"new_type": context.target_type}, context.target_type)
                if factors is not None:
                    return self.operation_cost_model.get_latency(factors, context.target_type)
            return reshard_step_cost(context.source_type, context.target_type)
        if realization != DistributedReshardRealization.SHARDED_VIEW:
            raise ValueError(f"Unknown distributed reshard realization {realization!r}.")

        # The entry caller owns completion and visibility of program outputs;
        # this alias has no downstream grid consumer inside the kernel.
        if context.usage_kind == DistributedReshardUsageKind.PROGRAM_OUTPUT:
            return 0
        source = context.source_type
        target = context.target_type
        if isinstance(source, DistributedType) and isinstance(target, DistributedType):
            if source.exclusive is not None and source.exclusive == target.exclusive:
                return 0
            if source.exclusive is not None or target.exclusive is not None:
                return self.grid_synchronization_cost
        if not isinstance(source, DistributedType):
            # Constants enter unified storage during materialization rather
            # than through a runtime producer/consumer edge.
            return 0
        if (
            isinstance(target, DistributedType)
            and is_local_shard_subview(source, target)
        ):
            return 0
        return self.grid_synchronization_cost

    def shared_publication_cost(self, context, realization) -> int:
        """The shareable portion of an internal read-only alias edge.

        A full-grid publication of an immutable SSA result serves all its
        read-only views, including views returned by the same invocation.
        A Partial reduction also needs that producer completion. Only its
        input barrier is shareable; transfer and arithmetic work are not.
        """
        if (
            context.source_kind is not DistributedReshardSourceKind.INTERNAL
            or context.usage_kind not in {
                DistributedReshardUsageKind.INTERNAL,
                DistributedReshardUsageKind.FUNCTION_BOUNDARY,
            }
            or not isinstance(context.source_type, DistributedType)
            or not isinstance(context.target_type, DistributedType)
            or context.source_type.exclusive != context.target_type.exclusive
        ):
            return 0
        if realization is DistributedReshardRealization.SHARDED_VIEW:
            return self.realization_cost(context, realization)
        if realization is DistributedReshardRealization.BOXING and self.operation_cost_model is not None:
            source = Node("<collective-source>", "builtin.var", (), context.source_type)
            factors = Boxing.cost_factors((source,), {"new_type": context.target_type}, context.target_type)
            if factors is not None and factors.grid_synchronizations == 1:
                return self.operation_cost_model.grid_synchronization_cycles
        return 0


__all__ = ["DistributedReshardCostModel"]
