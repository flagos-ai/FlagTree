# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Shared normalization type/value utilities; op definitions stay split."""

from __future__ import annotations

from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import TensorType, tensor_type
from triton.flagmega.ir.ops.tensors.pack import pack_physical
from triton.flagmega.ir.ops.tensors.unpack import unpack_physical
from triton.flagmega.ir.types import DType, DataType, VectorType


_FLOAT_DTYPES = frozenset({DType.BFLOAT16, DType.FLOAT32, DType.FLOAT8_E4M3FN})


def normalize_axis(axis: int, rank: int) -> int:
    value = axis + rank if axis < 0 else axis
    if value < 0 or value >= rank:
        raise IRSchemaError(f"Normalization axis {axis} is out of range for rank {rank}.")
    return value


def scalar_dtype(dtype: DataType) -> DType:
    return dtype.elem_type if isinstance(dtype, VectorType) else dtype


def is_float_dtype(dtype: DataType) -> bool:
    return scalar_dtype(dtype) in _FLOAT_DTYPES


def stats_tensor_type(value: TensorType, axis: int, use_mean: bool) -> TensorType:
    if value.rank == 0:
        raise IRSchemaError("NormStats input must have positive rank.")
    if not is_float_dtype(value.dtype):
        raise IRSchemaError("NormStats input must be floating point.")
    normalized = normalize_axis(axis, value.rank)
    shape = [2 if use_mean else 1]
    shape.extend(value.shape[index] if index < normalized else 1 for index in range(value.rank))
    return tensor_type(DType.FLOAT32, shape)


def unpack_default_vector(value, value_type: TensorType):
    if not isinstance(value_type.dtype, VectorType):
        return value
    if value_type.rank == 0:
        raise IRSchemaError("Normalization does not support a vector scalar tensor.")
    axes = (value_type.rank - 1,) * len(value_type.dtype.lanes)
    return unpack_physical(value, value_type.rank, value_type.dtype.lanes, axes)


def repack_default_vector(value, value_type: TensorType):
    if not isinstance(value_type.dtype, VectorType):
        return value
    axes = (value_type.rank - 1,) * len(value_type.dtype.lanes)
    return pack_physical(value, value_type.rank, value_type.dtype.lanes, axes)


def norm_stats_value(value, *, axis: int, use_mean: bool):
    normalized = normalize_axis(axis, value.ndim)
    reduction_axes = tuple(range(normalized, value.ndim))
    source = value.float()
    sum_sq = source.square().sum(dim=reduction_axes, keepdim=True)
    if not use_mean:
        return sum_sq.unsqueeze(0)
    import torch

    total = source.sum(dim=reduction_axes, keepdim=True)
    return torch.cat((total.unsqueeze(0), sum_sq.unsqueeze(0)), dim=0)


def norm_apply_value(
    value,
    stats,
    scale,
    bias,
    *,
    axis: int,
    epsilon: float,
    use_mean: bool,
    round_before_scale: bool = False,
):
    normalized = normalize_axis(axis, value.ndim)
    normalization_size = prod(int(extent) for extent in value.shape[normalized:])
    source = value.float()
    stats_value = stats.float()
    if use_mean:
        mean = stats_value[0] / normalization_size
        variance = stats_value[1] / normalization_size - mean.square()
        centered = source - mean
    else:
        variance = stats_value[0] / normalization_size
        centered = source
    rstd = (variance.clamp_min(0.0) + float(epsilon)).rsqrt()
    output = centered * rstd
    if round_before_scale:
        # This conversion is an observable IR boundary, not relaxed math.
        output = output.to(dtype=value.dtype).float()
    output = output * scale.float() + bias.float()
    return output.to(dtype=value.dtype)


__all__ = [
    "is_float_dtype",
    "norm_apply_value",
    "norm_stats_value",
    "normalize_axis",
    "repack_default_vector",
    "scalar_dtype",
    "stats_tensor_type",
    "unpack_default_vector",
]
