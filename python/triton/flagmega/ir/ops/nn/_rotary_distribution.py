# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Rotary partner ownership, independent of a device implementation."""

from functools import lru_cache
from itertools import product

from triton.flagmega.ir.distributed_type import SBPSplit
from triton.flagmega.ir.model import DistributedType, TensorType
from triton.flagmega.ir.local_shard import local_shard_descriptor
from triton.flagmega.ir.types import VectorType


def has_remote_rotary_pairs(value_type, rotary_dim=None, dimension_axis=-1):
    if not isinstance(value_type, DistributedType):
        return False
    policy = value_type.axis_policies[dimension_axis]
    if not isinstance(policy, SBPSplit):
        return False
    tensor = value_type.tensor
    lanes = tensor.dtype.lanes[0] if isinstance(tensor.dtype, VectorType) else 1
    extent = tensor.shape[dimension_axis]
    if not extent.is_fixed:
        return True
    return _has_remote_pairs(
        DistributedType(TensorType(tensor.dtype, (extent,)), (policy,), value_type.placement),
        lanes, extent.fixed_value * lanes if rotary_dim is None else rotary_dim,
    )


@lru_cache(maxsize=4096)
def _has_remote_pairs(dimension_type, lanes, rotary_dim):
    policy = dimension_type.axis_policies[0]
    split_axes = {axis for stage in policy.stages for axis in stage.hierarchy_axes}
    domains = tuple(range(count) if axis in split_axes else (0,)
                    for axis, count in enumerate(dimension_type.placement.hierarchy))
    owners = {}
    for owner in product(*domains):
        axis = local_shard_descriptor(dimension_type, owner).axes[0]
        for index in range(axis.active_extent.fixed_value):
            physical = axis.map_local_to_global(index).fixed_value
            for lane in range(lanes):
                scalar = physical * lanes + lane
                if scalar < rotary_dim:
                    owners[scalar] = owner
    half = rotary_dim // 2
    return any(owners[index] != owners[index + half] for index in range(half))
