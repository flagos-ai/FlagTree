# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Canonical per-owner view of a :class:`DistributedType`.

This is the Python counterpart of nncase's ``LocalShardDescriptor``.  It is a
derived IR contract, rather than a target schedule: ordinary kernels consume
the dense local domain while only layout transitions/collectives interpret
the relationship between owners.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.dim_expr import (
    Dimension,
    ceil_div,
    dim,
    dim_clamp,
    dim_max,
    dim_min,
)
from triton.flagmega.ir.distributed_type import (
    BlockCyclicSplit,
    ContiguousSplit,
    SBPBroadCast,
    SBPSplit,
    SplitStage,
)
from triton.flagmega.ir.model import DistributedType


@dataclass(frozen=True)
class ContiguousShardRegion:
    """One rectangular region of the logical tensor."""

    offset: tuple[Dimension, ...]
    shape: tuple[Dimension, ...]


@dataclass(frozen=True)
class LocalShardStageDescriptor:
    """One split stage bound to a placement coordinate."""

    stage: SplitStage
    parent_extent: Dimension
    local_capacity: Dimension
    active_extent: Dimension
    linear_shard_index: Dimension
    shard_count: int

    @property
    def is_contiguous(self) -> bool:
        distribution = self.stage.distribution
        if isinstance(distribution, ContiguousSplit):
            return True
        if isinstance(distribution, BlockCyclicSplit):
            maximum = self.local_capacity.maximum
            return maximum is not None and maximum <= distribution.block_size
        return False  # pragma: no cover - SplitDistribution is closed today.

    def map_local_to_parent(self, local_coordinate: int | str | Dimension) -> Dimension:
        """Map one dense local coordinate into this stage's parent domain."""

        coordinate = dim(local_coordinate)
        distribution = self.stage.distribution
        if isinstance(distribution, ContiguousSplit):
            return (
                self.linear_shard_index * self.local_capacity + coordinate
            ).simplify()
        if isinstance(distribution, BlockCyclicSplit):
            block = distribution.block_size
            return (
                (coordinate // block) * (self.shard_count * block)
                + self.linear_shard_index * block
                + coordinate % block
            ).simplify()
        raise IRSchemaError(  # pragma: no cover - SplitDistribution is closed today.
            f"Unsupported split distribution {type(distribution).__name__}."
        )


@dataclass(frozen=True)
class LocalShardAxisDescriptor:
    """Local capacity, active extent, and mapping for one tensor axis."""

    global_extent: Dimension
    local_capacity: Dimension
    active_extent: Dimension
    stages: tuple[LocalShardStageDescriptor, ...]

    @property
    def is_contiguous(self) -> bool:
        return all(stage.is_contiguous for stage in self.stages)

    @property
    def affine_stride(self) -> int | None:
        """Prove a constant global step throughout the active local domain.

        Compose the same staged map as ``map_local_to_global``. A cyclic
        stage is affine when its input step spans whole blocks, or when the
        complete input interval stays in one block. No sampling of endpoints
        is used: crossing a block boundary with an unaligned step is rejected.
        Empty and singleton domains admit any positive step.
        """

        count = self.active_extent.maximum
        origin, stride = dim(0), 1
        for stage in reversed(self.stages):
            distribution = stage.stage.distribution
            if isinstance(distribution, ContiguousSplit):
                origin = stage.map_local_to_parent(origin)
            elif isinstance(distribution, BlockCyclicSplit):
                block = distribution.block_size
                if stride % block == 0:
                    next_stride = stride * stage.shard_count
                elif (count is not None and origin.is_fixed
                      and origin.fixed_value // block
                      == (origin.fixed_value + max(count - 1, 0) * stride) // block):
                    next_stride = stride
                else:
                    return None
                origin = stage.map_local_to_parent(origin)
                stride = next_stride
            else:  # pragma: no cover - closed SplitDistribution union.
                return None
        return stride

    def map_local_to_global(self, local_coordinate: int | str | Dimension) -> Dimension:
        coordinate = dim(local_coordinate)
        for stage in reversed(self.stages):
            coordinate = stage.map_local_to_parent(coordinate)
        return coordinate.simplify()


@dataclass(frozen=True)
class LocalShardDescriptor:
    """Canonical dense local domain for one placement owner."""

    distributed_type: DistributedType
    coordinates: tuple[Dimension, ...]
    axes: tuple[LocalShardAxisDescriptor, ...]

    @property
    def local_capacity_shape(self) -> tuple[Dimension, ...]:
        return tuple(axis.local_capacity for axis in self.axes)

    @property
    def active_shape(self) -> tuple[Dimension, ...]:
        return tuple(axis.active_extent for axis in self.axes)

    @property
    def is_contiguous(self) -> bool:
        return all(axis.is_contiguous for axis in self.axes)

    @property
    def contiguous_region(self) -> ContiguousShardRegion | None:
        if not self.is_contiguous:
            return None
        return ContiguousShardRegion(
            tuple(axis.map_local_to_global(0) for axis in self.axes),
            self.active_shape,
        )

    @property
    def partial_axes(self) -> tuple[int, ...]:
        partial = self.distributed_type.partial
        return () if partial is None else tuple(partial.axes)

    @property
    def partial_group_coordinates(self) -> tuple[tuple[int, ...], ...] | None:
        """Fixed-coordinate collective group for a statically bound owner.

        Coordinates outside the partial axes remain fixed, so a Sum partial on
        mesh axis 0 of an 8x16 mesh forms 16 independent groups of eight; it is
        never accidentally reduced across all 128 owners.
        """

        partial_axes = self.partial_axes
        if not partial_axes:
            return ()
        fixed_coordinates: list[int] = []
        for coordinate in self.coordinates:
            if not coordinate.is_fixed:
                return None
            fixed_coordinates.append(coordinate.fixed_value)
        ranges = tuple(
            range(extent) if axis in partial_axes else (fixed_coordinates[axis],)
            for axis, extent in enumerate(
                self.distributed_type.placement.hierarchy
            )
        )
        return tuple(product(*ranges))


def local_shard_descriptor(
    distributed_type: DistributedType,
    coordinates: Sequence[int | str | Dimension],
) -> LocalShardDescriptor:
    """Bind staged SBP splits to one owner coordinate.

    ``local_capacity_shape`` is the uniform allocation shape. ``active_shape``
    is owner-specific and may be smaller at tails.  Mapping is retained as
    staged arithmetic for block-cyclic layouts instead of being expanded into
    an unmaintainable index list.
    """

    if not isinstance(distributed_type, DistributedType):
        raise TypeError("local_shard_descriptor requires a DistributedType.")
    placement = distributed_type.placement
    bound_coordinates = tuple(dim(value) for value in coordinates)
    if len(bound_coordinates) != placement.rank:
        raise IRSchemaError(
            f"Shard coordinate rank {len(bound_coordinates)} does not match "
            f"placement rank {placement.rank}."
        )
    for axis, (coordinate, extent) in enumerate(
        zip(bound_coordinates, placement.hierarchy)
    ):
        if coordinate.is_fixed and not 0 <= coordinate.fixed_value < extent:
            raise IRSchemaError(
                f"Shard coordinate {coordinate.fixed_value} on placement axis "
                f"{axis} is outside [0, {extent})."
            )

    axes: list[LocalShardAxisDescriptor] = []
    used_hierarchy_axes: set[int] = set()
    for global_extent, policy in zip(
        distributed_type.tensor.shape,
        distributed_type.axis_policies,
    ):
        if isinstance(policy, SBPBroadCast):
            axes.append(LocalShardAxisDescriptor(
                global_extent, global_extent, global_extent, ()
            ))
            continue
        if not isinstance(policy, SBPSplit):
            # Partial is represented separately on DistributedType in
            # FlagMega, so an axis policy can only be broadcast or split.
            axes.append(LocalShardAxisDescriptor(
                global_extent, global_extent, global_extent, ()
            ))
            continue

        capacity = global_extent
        for stage in policy.stages:
            capacity = _local_capacity(
                capacity,
                _stage_shard_count(stage, placement.hierarchy),
                stage,
            )

        parent_extent = global_extent
        active_extent = global_extent
        stages: list[LocalShardStageDescriptor] = []
        for stage in policy.stages:
            for hierarchy_axis in stage.hierarchy_axes:
                if hierarchy_axis in used_hierarchy_axes:
                    raise IRSchemaError(
                        f"Placement axis {hierarchy_axis} is assigned to more "
                        "than one tensor split policy."
                    )
                used_hierarchy_axes.add(hierarchy_axis)
            shard_count = _stage_shard_count(stage, placement.hierarchy)
            linear_shard_index = _linear_stage_coordinate(
                stage, bound_coordinates, placement.hierarchy
            )
            stage_capacity = _local_capacity(parent_extent, shard_count, stage)
            active_extent = _active_extent(
                parent_extent,
                stage_capacity,
                linear_shard_index,
                shard_count,
                stage,
            )
            stages.append(LocalShardStageDescriptor(
                stage,
                parent_extent,
                stage_capacity,
                active_extent,
                linear_shard_index,
                shard_count,
            ))
            # nncase intentionally feeds the active owner extent into the next
            # stage.  This preserves staged tail semantics.
            parent_extent = active_extent
        axes.append(LocalShardAxisDescriptor(
            global_extent,
            capacity.simplify(),
            active_extent.simplify(),
            tuple(stages),
        ))
    return LocalShardDescriptor(distributed_type, bound_coordinates, tuple(axes))


def unravel_placement_index(
    linear_index: int,
    hierarchy: Sequence[int],
) -> tuple[int, ...]:
    """Convert a row-major placement owner id to mesh coordinates."""

    owner_count = 1
    for extent in hierarchy:
        owner_count *= int(extent)
    if isinstance(linear_index, bool) or not 0 <= int(linear_index) < owner_count:
        raise IRSchemaError(
            f"Placement owner index {linear_index} is outside [0, {owner_count})."
        )
    remaining = int(linear_index)
    result = [0] * len(hierarchy)
    for axis in range(len(hierarchy) - 1, -1, -1):
        result[axis] = remaining % int(hierarchy[axis])
        remaining //= int(hierarchy[axis])
    return tuple(result)


def _stage_shard_count(stage: SplitStage, hierarchy: Sequence[int]) -> int:
    result = 1
    for axis in stage.hierarchy_axes:
        result *= int(hierarchy[axis])
    return result


def _linear_stage_coordinate(
    stage: SplitStage,
    coordinates: tuple[Dimension, ...],
    hierarchy: Sequence[int],
) -> Dimension:
    result = dim(0)
    for axis in stage.hierarchy_axes:
        result = result * int(hierarchy[axis]) + coordinates[axis]
    return result.simplify()


def _local_capacity(
    parent_extent: Dimension,
    shard_count: int,
    stage: SplitStage,
) -> Dimension:
    distribution = stage.distribution
    if isinstance(distribution, ContiguousSplit):
        if distribution.granularity is not None:
            return distribution.granularity
        return ceil_div(parent_extent, shard_count)
    if isinstance(distribution, BlockCyclicSplit):
        block = distribution.block_size
        return (ceil_div(parent_extent, shard_count * block) * block).simplify()
    raise IRSchemaError(  # pragma: no cover - SplitDistribution is closed today.
        f"Unsupported split distribution {type(distribution).__name__}."
    )


def _active_extent(
    parent_extent: Dimension,
    local_capacity: Dimension,
    linear_shard_index: Dimension,
    shard_count: int,
    stage: SplitStage,
) -> Dimension:
    # Match nncase's IsUniformCapacity fast path.  Besides simplifying codegen,
    # this proves that every valid owner has the same active extent without
    # leaving a coordinate-dependent clamp in the serialized ABI.
    if (
        parent_extent.is_fixed
        and local_capacity.is_fixed
        and parent_extent.fixed_value % shard_count == 0
        and parent_extent.fixed_value // shard_count
        == local_capacity.fixed_value
    ):
        return local_capacity
    distribution = stage.distribution
    if isinstance(distribution, ContiguousSplit):
        return dim_max(
            0,
            dim_min(
                local_capacity,
                parent_extent - linear_shard_index * local_capacity,
            ),
        ).simplify()
    if isinstance(distribution, BlockCyclicSplit):
        block = distribution.block_size
        cycle = shard_count * block
        return (
            (parent_extent // cycle) * block
            + dim_clamp(
                parent_extent % cycle - linear_shard_index * block,
                0,
                block,
            )
        ).simplify()
    raise IRSchemaError(  # pragma: no cover - SplitDistribution is closed today.
        f"Unsupported split distribution {type(distribution).__name__}."
    )


__all__ = [
    "ContiguousShardRegion",
    "LocalShardAxisDescriptor",
    "LocalShardDescriptor",
    "LocalShardStageDescriptor",
    "local_shard_descriptor",
    "unravel_placement_index",
]
