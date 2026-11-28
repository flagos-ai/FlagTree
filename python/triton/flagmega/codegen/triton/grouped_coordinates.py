# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Prove group uniformity from logical shard maps, independent of storage."""

from itertools import product

from triton.flagmega.ir import (
    BlockCyclicSplit,
    ContiguousSplit,
    DistributedType,
    TensorType,
    VectorType,
    local_shard_descriptor,
    logical_type,
)


def tiles_stay_within_groups(value_type, axis: int, tile: int, group_size: int) -> bool:
    """Every active tile lies in one contiguous logical group on ``axis``.

    Every stage must preserve a unit step within each aligned tile. A cyclic
    block whose size is a multiple of the tile preserves that property even
    when the complete shard is non-affine. Packed lane coordinates need a
    separate scalar-coordinate proof.
    """
    tensor = logical_type(value_type)
    if (not isinstance(tensor, TensorType) or isinstance(tensor.dtype, VectorType)
            or not -tensor.rank <= axis < tensor.rank or tile <= 0 or group_size <= 0 or group_size % tile):
        return False
    if not tensor.shape[axis].is_fixed:
        return False
    if not isinstance(value_type, DistributedType):
        return True
    for owner in product(*(range(size) for size in value_type.placement.hierarchy)):
        descriptor = local_shard_descriptor(value_type, owner).axes[axis]
        origin = descriptor.map_local_to_global(0)
        if not origin.is_fixed or origin.fixed_value % tile:
            return False
        if descriptor.affine_stride != 1:
            for stage in descriptor.stages:
                distribution = stage.stage.distribution
                if isinstance(distribution, BlockCyclicSplit):
                    if distribution.block_size % tile:
                        return False
                elif isinstance(distribution, ContiguousSplit):
                    shift = (stage.linear_shard_index * stage.local_capacity).simplify()
                    if not shift.is_fixed or shift.fixed_value % tile:
                        return False
                else:
                    return False
    return True


__all__ = ["tiles_stay_within_groups"]
