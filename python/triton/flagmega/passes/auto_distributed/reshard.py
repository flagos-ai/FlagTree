# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Bounded reshard-path planner ported from nncase."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Callable

from triton.flagmega.ir import DistributedType, IRType, SBP, TensorType
from triton.flagmega.passes.auto_distributed.reshard_decomposition import (
    get_partial_reduce_scatter_intermediates,
)


@dataclass(frozen=True)
class DistributedReshardPlan:
    step_types: tuple[IRType, ...]


def can_box(source_type: IRType, target_type: IRType) -> bool:
    source_tensor = source_type.tensor if isinstance(source_type, DistributedType) else source_type
    target_tensor = target_type.tensor if isinstance(target_type, DistributedType) else target_type
    if source_tensor != target_tensor:
        return False
    if isinstance(source_type, DistributedType) and isinstance(target_type, DistributedType):
        return source_type.placement == target_type.placement
    return True


class DistributedReshardPlanner:
    default_max_hops = 3

    @classmethod
    def plan(
        cls,
        source_type: IRType,
        target_type: IRType,
        predicate: Callable[[IRType, IRType], bool] = can_box,
        max_hops: int = default_max_hops,
    ) -> tuple[DistributedReshardPlan, ...]:
        if max_hops < 1:
            return ()
        plans: list[DistributedReshardPlan] = []
        seen: set[tuple[IRType, ...]] = set()

        def add(*steps: IRType) -> None:
            normalized: list[IRType] = []
            previous = source_type
            for step in steps:
                if step != previous:
                    normalized.append(step)
                    previous = step
            key = tuple(normalized)
            previous = source_type
            if not key or len(key) > max_hops or key in seen:
                return
            if all(predicate(lhs, rhs) for lhs, rhs in zip((source_type, *key[:-1]), key)):
                seen.add(key)
                plans.append(DistributedReshardPlan(key))

        add(target_type)
        if (
            max_hops == 1
            or not isinstance(source_type, DistributedType)
            or not isinstance(target_type, DistributedType)
            or source_type.tensor != target_type.tensor
            or source_type.placement != target_type.placement
        ):
            return tuple(plans)
        for intermediate in get_partial_reduce_scatter_intermediates(source_type, target_type):
            add(intermediate, target_type)
        if plans:
            return tuple(plans)
        source_no_partial = DistributedType(
            source_type.tensor, source_type.axis_policies, source_type.placement)
        broadcast = DistributedType(
            source_type.tensor,
            tuple(SBP.broadcast() for _ in source_type.tensor.shape),
            source_type.placement,
        )
        add(source_no_partial, target_type)
        add(broadcast, target_type)
        add(source_no_partial, broadcast, target_type)
        return tuple(plans)


def reshard_step_cost(source_type: IRType, target_type: IRType) -> int:
    if source_type == target_type:
        return 0
    if not isinstance(source_type, DistributedType) and isinstance(target_type, DistributedType):
        return 1
    tensor = source_type.tensor if isinstance(source_type, DistributedType) else source_type
    if not isinstance(tensor, TensorType) or any(not dimension.is_fixed for dimension in tensor.shape):
        return 100_000_000
    size = prod(dimension.fixed_value for dimension in tensor.shape) * tensor.dtype.itemsize
    return min(max(size * 100, 1), 2_000_000_000)


def reshard_plan_cost(source_type: IRType, plan: DistributedReshardPlan) -> int:
    total = 0
    previous = source_type
    for step in plan.step_types:
        total = min(total + reshard_step_cost(previous, step), 2_000_000_000)
        previous = step
    return total


__all__ = [
    "DistributedReshardPlan",
    "DistributedReshardPlanner",
    "can_box",
    "reshard_plan_cost",
    "reshard_step_cost",
]
