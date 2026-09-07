# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Shared distributed type-inference primitives used by op definitions."""

from __future__ import annotations

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_type import Placement, SBP, SBPBroadCast, SBPSplit
from triton.flagmega.ir.model import DistributedType, IRType, TensorType, TupleType


def tensor_of(value: IRType) -> TensorType:
    result = value.tensor if isinstance(value, DistributedType) else value
    if not isinstance(result, TensorType):
        raise IRSchemaError(f"Expected a tensor-like type, got {type(value).__name__}.")
    return result


def broadcast_type(tensor: TensorType, placement: Placement) -> DistributedType:
    return DistributedType(tensor, tuple(SBP.broadcast() for _ in tensor.shape), placement)


def broadcast_ir_type(value: IRType, placement: Placement) -> IRType:
    """Build nncase's recursively broadcast distributed candidate type.

    Tensor leaves become ``DistributedType`` values, tuple structure is
    retained, and non-tensor leaves such as state/reference handles remain
    standalone.  AutoDistributed uses this both for the replicated candidate
    of a call and for the program-output contract.
    """

    if isinstance(value, DistributedType):
        return broadcast_type(value.tensor, placement)
    if isinstance(value, TensorType):
        return broadcast_type(value, placement)
    if isinstance(value, TupleType):
        return TupleType(tuple(broadcast_ir_type(field, placement) for field in value.fields))
    return value


def split_type(
    tensor: TensorType,
    tensor_axis: int,
    placement: Placement,
    hierarchy_axes: tuple[int, ...] = (0,),
    *,
    block_size: int | None = None,
) -> DistributedType:
    if tensor_axis < 0:
        tensor_axis += tensor.rank
    policies = [SBP.broadcast() for _ in tensor.shape]
    policies[tensor_axis] = (
        SBP.split_contiguous(hierarchy_axes)
        if block_size is None
        else SBP.split_block_cyclic(hierarchy_axes, block_size)
    )
    return DistributedType(tensor, tuple(policies), placement)


def placement_of(*values: IRType) -> Placement | None:
    placements = {value.placement for value in values if isinstance(value, DistributedType)}
    if len(placements) > 1:
        raise IRSchemaError("Distributed operands must use one placement.")
    return next(iter(placements), None)


def split_policy(value: IRType, tensor_axis: int) -> SBPSplit | None:
    if not isinstance(value, DistributedType):
        return None
    if tensor_axis < 0:
        tensor_axis += value.tensor.rank
    policy = value.axis_policies[tensor_axis]
    return policy if isinstance(policy, SBPSplit) else None


def all_broadcast(value: IRType) -> bool:
    return isinstance(value, DistributedType) and value.partial is None and all(
        isinstance(policy, SBPBroadCast) for policy in value.axis_policies)


def broadcast_placement_of(*values: IRType) -> Placement | None:
    """Return the common placement when every distributed leaf is broadcast.

    ``None`` means either that no distributed leaf exists or that at least one
    leaf carries a split/partial policy and therefore cannot use generic
    broadcast lifting.
    """

    placements: set[Placement] = set()
    saw_distributed = False

    def visit(value: IRType) -> bool:
        nonlocal saw_distributed
        if isinstance(value, DistributedType):
            saw_distributed = True
            if not all_broadcast(value):
                return False
            placements.add(value.placement)
            return True
        if isinstance(value, TupleType):
            return all(visit(field) for field in value.fields)
        return True

    if not all(visit(value) for value in values) or not saw_distributed:
        return None
    if len(placements) != 1:
        raise IRSchemaError("Broadcast distributed operands must use one placement.")
    return next(iter(placements))


__all__ = [
    "all_broadcast",
    "broadcast_ir_type",
    "broadcast_placement_of",
    "broadcast_type",
    "placement_of",
    "split_policy",
    "split_type",
    "tensor_of",
]
