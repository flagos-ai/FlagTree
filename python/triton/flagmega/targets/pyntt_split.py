# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""PyNTT block-cyclic split candidates, independent of a vendor machine."""

from __future__ import annotations

from triton.flagmega.ir import SBP, SBPSplit, SplitStage
from triton.flagmega.passes.auto_distributed.split_candidates import (
    DistributedSplitCandidateContext,
)


class PyNttDistributedSplitCandidateProvider:
    """Port of nncase ``PyNTTDistributedSplitCandidateProvider``."""

    def __init__(self, block_bytes: int = 128) -> None:
        if (
            isinstance(block_bytes, bool)
            or block_bytes <= 0
            or block_bytes & (block_bytes - 1)
        ):
            raise ValueError(
                "PyNTT block-cyclic byte granularity must be a positive power of two."
            )
        self.block_bytes = int(block_bytes)

    @property
    def identity(self) -> str:
        return f"pyntt-block-cyclic/v1(block_bytes={self.block_bytes})"

    def get_candidates(
        self,
        context: DistributedSplitCandidateContext,
    ) -> tuple[SBPSplit, ...]:
        axes = context.hierarchy_axes
        if not axes or any(axis < 0 or axis >= context.placement.rank for axis in axes):
            raise ValueError("PyNTT split candidate hierarchy axes are invalid.")
        contiguous = SBP.split_contiguous(axes, context.contiguous_granularity)
        groups = _group_physical_levels(context)
        if not any(level == "b" for level, _ in groups):
            return (contiguous,)
        parent_extent = context.maximum_extent
        if parent_extent is None or parent_extent <= 0:
            return (contiguous,)
        stages: list[SplitStage] = []
        for level, group_axes in groups:
            shard_count = _product(
                context.placement.hierarchy[axis] for axis in group_axes
            )
            if level == "b":
                maximum_useful = max(1, parent_extent // shard_count)
                preferred = max(1, self.block_bytes // context.tensor_type.dtype.itemsize)
                block_size = _highest_power_of_two_at_most(
                    min(maximum_useful, preferred)
                )
                stages.append(SplitStage.block_cyclic(group_axes, block_size))
                parent_extent = (
                    (parent_extent + shard_count * block_size - 1)
                    // (shard_count * block_size)
                    * block_size
                )
            else:
                stages.append(SplitStage.contiguous(group_axes))
                parent_extent = (parent_extent + shard_count - 1) // shard_count
        block_cyclic = SBP.split(*stages)
        return (block_cyclic,) if block_cyclic == contiguous else (
            block_cyclic,
            contiguous,
        )


def _group_physical_levels(
    context: DistributedSplitCandidateContext,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    groups: list[tuple[str, tuple[int, ...]]] = []
    for axis in context.hierarchy_axes:
        level = context.placement.hierarchy_levels[axis]
        if groups and groups[-1][0] == level:
            groups[-1] = (level, (*groups[-1][1], axis))
        else:
            groups.append((level, (axis,)))
    return tuple(groups)


def _highest_power_of_two_at_most(value: int) -> int:
    return 1 << (value.bit_length() - 1)


def _product(values) -> int:
    result = 1
    for value in values:
        result *= value
    return result


__all__ = ["PyNttDistributedSplitCandidateProvider"]
