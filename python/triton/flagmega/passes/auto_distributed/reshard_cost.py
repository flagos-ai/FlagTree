# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Physical costs for materializing semantic distributed reshard edges."""

from __future__ import annotations

from dataclasses import dataclass

from triton.flagmega.ir import DistributedType, is_local_shard_subview
from triton.flagmega.passes.auto_distributed.realization import (
    DistributedReshardRealization,
    DistributedReshardRealizationContext,
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
            return reshard_step_cost(context.source_type, context.target_type)
        if realization != DistributedReshardRealization.SHARDED_VIEW:
            raise ValueError(f"Unknown distributed reshard realization {realization!r}.")

        # The entry caller owns completion and visibility of program outputs;
        # this alias has no downstream grid consumer inside the kernel.
        if context.usage_kind == DistributedReshardUsageKind.PROGRAM_OUTPUT:
            return 0
        source = context.source_type
        target = context.target_type
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


__all__ = ["DistributedReshardCostModel"]
