# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Per-access barrier coverage, following nncase PendingEffectSet.BarrierCoverage."""

from dataclasses import dataclass, field

from triton.flagmega.ir import DistributedType, Placement


SynchronizationRequirement = tuple[str, tuple[int, ...], Placement | None]


@dataclass
class BarrierCoverage:
    block_synchronized: bool = False
    full_chip_synchronized: bool = False
    placement: Placement | None = None
    axis_group_axes: set[int] = field(default_factory=set)

    def covers(self, requirement: SynchronizationRequirement) -> bool:
        scope, axes, placement = requirement
        if scope == "block":
            return self.block_synchronized
        if self.full_chip_synchronized:
            return True
        return bool(axes) and placement is not None and placement == self.placement and self.axis_group_axes.issuperset(axes)

    def apply(self, requirement: SynchronizationRequirement, distributed_type: DistributedType | None) -> None:
        scope, axes, placement = requirement
        self.block_synchronized = True
        if scope == "block":
            return
        if not axes:
            self.full_chip_synchronized = True
            return
        placement = placement or (None if distributed_type is None else distributed_type.placement)
        if placement is None or (
            distributed_type is not None and distributed_type.placement != placement
        ) or (self.placement is not None and self.placement != placement):
            return
        self.placement = placement
        self.axis_group_axes.update(axes)
        block_axes = {
            index for index, level in enumerate(placement.hierarchy_levels) if level == "b"
        }
        self.full_chip_synchronized = self.axis_group_axes.issuperset(block_axes)
