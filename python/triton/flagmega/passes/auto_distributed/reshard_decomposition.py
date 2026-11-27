# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Target-independent intermediate types for decomposed distributed reshards."""

from __future__ import annotations

from triton.flagmega.ir import (
    DistributedType,
    SBP,
    SBPBroadCast,
    SBPPartial,
    SBPSplit,
    ceil_div,
    is_distributable,
)


def get_partial_reduce_scatter_intermediates(
    source: DistributedType,
    target: DistributedType,
) -> tuple[DistributedType, ...]:
    """Port nncase's ``DistributedReshardDecomposition`` exactly at type level.

    A partial-to-materialized transition may discharge some placement axes
    directly into the target split.  Any remaining partial axes can first be
    reduce-scattered into one tensor axis which remains broadcast at both
    endpoints.  Direct all-reduce and this decomposed program are deliberately
    both returned to the global extractor.
    """

    partial = source.partial
    if (
        partial is None
        or target.partial is not None
        or source.tensor != target.tensor
        or source.placement != target.placement
    ):
        return ()
    remaining = _validate_partial_transition(source, target, partial)
    if not remaining:
        return ()
    divisor = 1
    for placement_axis in remaining:
        divisor *= source.placement.hierarchy[placement_axis]
    values: list[DistributedType] = []
    for tensor_axis, (source_policy, target_policy) in enumerate(
        zip(source.axis_policies, target.axis_policies)
    ):
        if not isinstance(source_policy, SBPBroadCast) or not isinstance(target_policy, SBPBroadCast):
            continue
        policies = list(target.axis_policies)
        policies[tensor_axis] = SBP.split_contiguous(
            remaining,
            ceil_div(target.tensor.shape[tensor_axis], divisor),
        )
        candidate_policies = tuple(policies)
        # The nncase decomposition probes every broadcast tensor axis.  Its
        # graph can temporarily hold candidates which are rejected later by
        # DistributedUtility.IsDistributable; FlagMega validates at
        # DistributedType construction, so perform the same filter here.
        if is_distributable(target.tensor, candidate_policies, target.placement):
            values.append(DistributedType(target.tensor, candidate_policies, target.placement))
    return tuple(values)


def _validate_partial_transition(
    source: DistributedType,
    target: DistributedType,
    partial: SBPPartial,
) -> tuple[int, ...]:
    partial_axes = set(partial.axes)
    if not partial_axes or any(axis < 0 or axis >= source.placement.rank for axis in partial_axes):
        return ()
    target_split_axes: set[int] = set()
    for source_policy, target_policy in zip(source.axis_policies, target.axis_policies):
        if isinstance(source_policy, SBPBroadCast) and isinstance(target_policy, SBPBroadCast):
            continue
        if isinstance(source_policy, SBPBroadCast) and isinstance(target_policy, SBPSplit):
            axes = target_policy.hierarchy_axes
            if not all(axis in partial_axes for axis in axes) or any(axis in target_split_axes for axis in axes):
                return ()
            target_split_axes.update(axes)
            continue
        if isinstance(source_policy, SBPSplit) and source_policy == target_policy:
            axes = source_policy.hierarchy_axes
            if any(axis in target_split_axes for axis in axes):
                return ()
            target_split_axes.update(axes)
            continue
        return ()
    return tuple(axis for axis in partial.axes if axis not in target_split_axes)


__all__ = ["get_partial_reduce_scatter_intermediates"]
