# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Prove that each MLA owner receives complete, matching per-head row groups."""

from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_type import (
    BlockCyclicSplit, ContiguousSplit, SBP, SBPBroadCast, SBPSplit, SplitStage, scale_split_units,
)
from triton.flagmega.ir.model import DistributedType, NoneType


def _explicit_split(policy, extent, placement):
    if isinstance(policy, SBPBroadCast):
        return policy
    if not isinstance(policy, SBPSplit):
        raise IRSchemaError("MLA projection permits only broadcast or split head ownership.")
    stages = []
    for stage in policy.stages:
        owners = prod(placement.hierarchy[axis] for axis in stage.hierarchy_axes)
        distribution = stage.distribution
        if isinstance(distribution, ContiguousSplit):
            extent = ((extent + owners - 1) // owners if distribution.granularity is None
                      else distribution.granularity.fixed_value)
            stages.append(SplitStage.contiguous(stage.hierarchy_axes, extent))
        elif isinstance(distribution, BlockCyclicSplit):
            block = distribution.block_size
            extent = ((extent + block * owners - 1) // (block * owners)) * block
            stages.append(stage)
    return SBP.split(*stages)


def validate_mla_head_distribution(value, weight, scale, attrs):
    """Keep tokens/features complete and correlate head/weight/scale owners.

    Implicit contiguous units are expanded *before* multiplication by row-group
    width. Comparing implicit policies directly would accept different owner
    maps whenever a group's scaled extent rounds differently.
    """
    operands = tuple(t for t in (value, weight, scale) if not isinstance(t, NoneType))
    mode = isinstance(value, DistributedType)
    if any(isinstance(t, DistributedType) != mode for t in operands):
        raise IRSchemaError("MLA projection tensor operands must share one distribution.")
    if not mode:
        return
    for t in operands:
        if (t.placement != value.placement or t.partial is not None or t.exclusive is not None
                or any(not isinstance(p, (SBPBroadCast, SBPSplit)) for p in t.axis_policies)):
            raise IRSchemaError("MLA projection requires one placement and complete, nonpartial head groups.")
    if (any(not isinstance(value.axis_policies[a], SBPBroadCast) for a in (0, 2))
            or not isinstance(weight.axis_policies[1], SBPBroadCast)):
        raise IRSchemaError("MLA projection token and reduction/feature axes must be broadcast.")
    heads = value.tensor.shape[1].fixed_value
    group = attrs['qk_head_dim'] + attrs['value_head_dim']
    head = _explicit_split(value.axis_policies[1], heads, value.placement)
    expected = scale_split_units(head, group, 1) if isinstance(head, SBPSplit) else head
    actual = _explicit_split(weight.axis_policies[0], heads * group, value.placement)
    if expected != actual:
        raise IRSchemaError("MLA projection weight rows must have the same complete head owners as value.")
    if isinstance(scale, NoneType):
        return
    if not isinstance(scale.axis_policies[1], SBPBroadCast):
        raise IRSchemaError("MLA projection scale columns must be broadcast.")
    if isinstance(scale.axis_policies[0], SBPBroadCast):
        return
    block_n = attrs['weight_block_n']
    if group % block_n:
        raise IRSchemaError("MLA projection sharded scales require each head row group to align to scale blocks.")
    expected = scale_split_units(head, group // block_n, 1) if isinstance(head, SBPSplit) else head
    actual = _explicit_split(scale.axis_policies[0], heads * group // block_n, value.placement)
    if expected != actual:
        raise IRSchemaError("MLA projection scales must be broadcast or match complete head row owners.")


def mla_head_distribution_choices(choices, attrs):
    # Reject incompatible value/weight pairs before considering scale choices.
    # This retains correlated head shards instead of emitting a cartesian set
    # that asks inference to repair mismatched owners through Boxing.
    for value in choices[0]:
        for weight in choices[1]:
            try:
                validate_mla_head_distribution(value, weight, NoneType(), attrs)
            except IRSchemaError:
                continue
            for scale in choices[2]:
                try:
                    validate_mla_head_distribution(value, weight, scale, attrs)
                except IRSchemaError:
                    continue
                yield value, weight, scale
