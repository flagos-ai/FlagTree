# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Shared type/value utilities for explicit paged-attention split states."""

from __future__ import annotations

from collections.abc import Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import (
    ReduceOp,
    SBPBroadCast,
    SBPPartial,
    SBPSplit,
)
from triton.flagmega.ir.model import (
    DistributedType,
    IRType,
    TensorType,
    TupleType,
    tensor_type,
)
from triton.flagmega.ir.types import DType, DataType, VectorType


def create_partial_state_type(
    attention_type: IRType,
    layout: Sequence[str],
    hidden_size: int,
    split_hierarchy_axis: int,
    split_count: int,
) -> TupleType:
    """Create nncase's FP32 max/sum/acc partial-state tuple."""

    _require_split_count(split_hierarchy_axis, split_count)
    head_axis, dim_axis = _layout_axes(layout)
    attention = tensor_of(attention_type)
    scalar_shape = list(attention.shape)
    scalar_shape[dim_axis] = (
        scalar_shape[dim_axis] * vector_lane_count(attention.dtype)
    ).simplify()
    _require_hidden_size(scalar_shape, head_axis, dim_axis, hidden_size)
    stats_shape = list(scalar_shape)
    stats_shape[dim_axis] = 1
    stats_type = tensor_type(DType.FLOAT32, stats_shape)
    acc_type = tensor_type(DType.FLOAT32, scalar_shape)
    if isinstance(attention_type, TensorType):
        return TupleType((stats_type, stats_type, acc_type))
    if not isinstance(attention_type, DistributedType):
        raise IRSchemaError(
            "PagedAttentionPartial requires a tensor or distributed tensor result."
        )
    placement = attention_type.placement
    if (
        attention_type.partial is not None
        or len(attention_type.axis_policies) != attention.rank
        or any(isinstance(policy, SBPPartial) for policy in attention_type.axis_policies)
        or split_hierarchy_axis >= placement.rank
        or not placement.is_physical_block_axis(split_hierarchy_axis)
        or placement.hierarchy[split_hierarchy_axis] != split_count
        or any(
            uses_hierarchy_axis(policy, split_hierarchy_axis)
            for policy in attention_type.axis_policies
        )
    ):
        raise IRSchemaError(
            "PagedAttentionPartial requires an unused physical block hierarchy axis "
            f"{split_hierarchy_axis} with extent {split_count}, got {attention_type!r}."
        )
    policies = attention_type.axis_policies
    return TupleType((
        DistributedType(
            stats_type,
            policies,
            placement,
            SBPPartial((split_hierarchy_axis,), ReduceOp.MAX),
        ),
        DistributedType(
            stats_type,
            policies,
            placement,
            SBPPartial((split_hierarchy_axis,), ReduceOp.SUM),
        ),
        DistributedType(
            acc_type,
            policies,
            placement,
            SBPPartial((split_hierarchy_axis,), ReduceOp.SUM),
        ),
    ))


def create_combine_output_type(
    max_state: IRType,
    sum_state: IRType,
    acc_state: IRType,
    layout: Sequence[str],
    hidden_size: int,
    output_data_type: DataType,
    output_type: IRType,
    split_hierarchy_axis: int,
    split_count: int,
) -> IRType:
    """Validate and discharge P(Max)/P(Sum)/P(Sum) into ``output_type``."""

    _require_split_count(split_hierarchy_axis, split_count)
    head_axis, dim_axis = _layout_axes(layout)
    max_tensor = tensor_of(max_state)
    sum_tensor = tensor_of(sum_state)
    acc_tensor = tensor_of(acc_state)
    if (
        max_tensor.dtype != DType.FLOAT32
        or sum_tensor.dtype != DType.FLOAT32
        or acc_tensor.dtype != DType.FLOAT32
        or max_tensor.rank != len(layout)
        or sum_tensor != max_tensor
        or acc_tensor.rank != max_tensor.rank
        or not max_tensor.shape[dim_axis].is_fixed
        or max_tensor.shape[dim_axis].fixed_value != 1
        or any(
            max_tensor.shape[axis] != acc_tensor.shape[axis]
            for axis in range(max_tensor.rank)
            if axis != dim_axis
        )
    ):
        raise IRSchemaError(
            "PagedAttentionCombine requires compatible FP32 max, sum, and accumulator state tensors."
        )
    lanes = vector_lane_count(output_data_type)
    scalar_dim = acc_tensor.shape[dim_axis]
    if not scalar_dim.is_fixed or scalar_dim.fixed_value % lanes:
        raise IRSchemaError(
            f"PagedAttentionCombine Dim extent {scalar_dim} is not divisible by "
            f"output lanes {lanes}."
        )
    output_shape = list(acc_tensor.shape)
    output_shape[dim_axis] = scalar_dim.fixed_value // lanes
    _require_hidden_size(output_shape, head_axis, dim_axis, hidden_size, lanes)
    output_tensor = tensor_type(output_data_type, output_shape)
    if all(isinstance(value, TensorType) for value in (max_state, sum_state, acc_state)):
        if output_type != output_tensor:
            raise IRSchemaError(
                f"PagedAttentionCombine output contract {output_type!r} does not match "
                f"{output_tensor!r}."
            )
        return output_tensor
    if not all(
        isinstance(value, DistributedType)
        for value in (max_state, sum_state, acc_state)
    ):
        raise IRSchemaError(
            "PagedAttentionCombine requires either three logical or three distributed states."
        )
    distributed_max = max_state
    distributed_sum = sum_state
    distributed_acc = acc_state
    assert isinstance(distributed_max, DistributedType)
    assert isinstance(distributed_sum, DistributedType)
    assert isinstance(distributed_acc, DistributedType)
    placement = distributed_max.placement
    max_partial = distributed_max.partial
    sum_partial = distributed_sum.partial
    acc_partial = distributed_acc.partial
    if (
        distributed_sum.placement != placement
        or distributed_acc.placement != placement
        or not isinstance(max_partial, SBPPartial)
        or max_partial.reduce_op is not ReduceOp.MAX
        or not isinstance(sum_partial, SBPPartial)
        or sum_partial.reduce_op is not ReduceOp.SUM
        or not isinstance(acc_partial, SBPPartial)
        or acc_partial.reduce_op is not ReduceOp.SUM
        or max_partial.axes != (split_hierarchy_axis,)
        or sum_partial.axes != max_partial.axes
        or acc_partial.axes != max_partial.axes
        or distributed_sum.axis_policies != distributed_max.axis_policies
        or distributed_acc.axis_policies != distributed_max.axis_policies
        or len(distributed_max.axis_policies) != max_tensor.rank
        or any(
            uses_hierarchy_axis(policy, split_hierarchy_axis)
            for policy in distributed_max.axis_policies
        )
        or split_hierarchy_axis >= placement.rank
        or placement.hierarchy[split_hierarchy_axis] != split_count
    ):
        raise IRSchemaError(
            "PagedAttentionCombine requires matching FP32 P(Max)/P(Sum)/P(Sum) "
            "states on one placement."
        )
    if (
        not isinstance(output_type, DistributedType)
        or output_type.tensor != output_tensor
        or output_type.placement != placement
        or output_type.partial is not None
        or len(output_type.axis_policies) != output_tensor.rank
        or any(isinstance(policy, SBPPartial) for policy in output_type.axis_policies)
        or not can_combine_to(
            distributed_acc, output_type, split_hierarchy_axis
        )
    ):
        raise IRSchemaError(
            f"PagedAttentionCombine cannot discharge P([{split_hierarchy_axis}]) "
            f"into {output_type!r}."
        )
    return output_type


def can_combine_to(
    partial_state: DistributedType,
    output_type: DistributedType,
    split_hierarchy_axis: int,
) -> bool:
    """Port nncase ``PagedAttentionSplitTypeUtility.CanCombineTo``."""

    if (
        partial_state.placement != output_type.placement
        or partial_state.partial is None
        or partial_state.partial.axes != (split_hierarchy_axis,)
        or output_type.partial is not None
        or len(partial_state.axis_policies) != len(output_type.axis_policies)
    ):
        return False
    input_hierarchy = hierarchy_axis_policies(
        partial_state.axis_policies, partial_state.placement.rank
    )
    output_hierarchy = hierarchy_axis_policies(
        output_type.axis_policies, output_type.placement.rank
    )
    if input_hierarchy is None or output_hierarchy is None:
        return False
    for axis, (source, target) in enumerate(zip(input_hierarchy, output_hierarchy)):
        if axis == split_hierarchy_axis:
            if source is not None:
                return False
            continue
        if source != target:
            return False
    return True


def hierarchy_axis_policies(
    policies: Sequence[object], placement_rank: int
) -> tuple[int | None, ...] | None:
    result: list[int | None] = [None] * placement_rank
    for tensor_axis, policy in enumerate(policies):
        if isinstance(policy, SBPBroadCast):
            continue
        if not isinstance(policy, SBPSplit):
            return None
        for hierarchy_axis in policy.hierarchy_axes:
            if hierarchy_axis >= placement_rank or result[hierarchy_axis] is not None:
                return None
            result[hierarchy_axis] = tensor_axis
    return tuple(result)


def uses_hierarchy_axis(policy: object, hierarchy_axis: int) -> bool:
    if isinstance(policy, SBPSplit):
        return hierarchy_axis in policy.hierarchy_axes
    if isinstance(policy, SBPPartial):
        return hierarchy_axis in policy.axes
    return False


def vector_lane_count(dtype: DataType) -> int:
    return dtype.lane_count if isinstance(dtype, VectorType) else 1


def unpack_dim_to_scalar(value, value_type: TensorType, dim_axis: int):
    if not isinstance(value_type.dtype, VectorType):
        return value
    from triton.flagmega.ir.ops.tensors.unpack import unpack_physical

    return unpack_physical(
        value,
        value_type.rank,
        value_type.dtype.lanes,
        (dim_axis,) * len(value_type.dtype.lanes),
    )


def pack_dim_from_scalar(value, dtype: DataType, dim_axis: int):
    if not isinstance(dtype, VectorType):
        return value
    from triton.flagmega.ir.ops.tensors.pack import pack_physical

    return pack_physical(
        value,
        value.ndim,
        dtype.lanes,
        (dim_axis,) * len(dtype.lanes),
    )


def _layout_axes(layout: Sequence[str]) -> tuple[int, int]:
    normalized = tuple(layout)
    if len(normalized) != 3 or set(normalized) != {"seq", "head", "dim"}:
        raise IRSchemaError(
            "Paged attention requires one seq, head, and dim layout axis."
        )
    return normalized.index("head"), normalized.index("dim")


def _require_split_count(axis: int, count: int) -> None:
    if isinstance(axis, bool) or not isinstance(axis, int) or axis < 0:
        raise IRSchemaError(
            "Paged-attention split_hierarchy_axis must be a non-negative integer."
        )
    if isinstance(count, bool) or not isinstance(count, int) or count <= 1:
        raise IRSchemaError("Paged-attention split_count must be greater than one.")


def _require_hidden_size(
    shape: Sequence[object],
    head_axis: int,
    dim_axis: int,
    hidden_size: int,
    lanes: int = 1,
) -> None:
    hidden = (shape[head_axis] * shape[dim_axis] * lanes).simplify()
    if hidden.is_fixed and hidden.fixed_value != hidden_size:
        raise IRSchemaError(
            f"Paged-attention hidden extent {hidden.fixed_value} does not match "
            f"hidden_size {hidden_size}."
        )


__all__ = [
    "can_combine_to",
    "create_combine_output_type",
    "create_partial_state_type",
    "pack_dim_from_scalar",
    "unpack_dim_to_scalar",
    "uses_hierarchy_axis",
    "vector_lane_count",
]
