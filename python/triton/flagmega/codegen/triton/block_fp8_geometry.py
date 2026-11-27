# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Prove the fixed chip-local row domain of an ordered FP8 projection."""

from dataclasses import dataclass
from math import prod

from triton.flagmega.ir import (
    SBP,
    DistributedType,
    DType,
    VectorType,
    local_tensor_type,
    logical_type,
    scale_split_units,
)
from triton.flagmega.ir.distributed_type import ContiguousSplit, SBPSplit


@dataclass(frozen=True)
class CooperativeBlockGeometry:
    local_rows: int
    reduction: int
    hierarchy: tuple[int, ...]
    levels: str
    row_axes: tuple[int, ...]

    @property
    def chip_rows(self):
        return self.local_rows * prod(self.hierarchy[i] for i in self.row_axes)

    @property
    def k_blocks(self):
        return self.reduction // 128

    @property
    def row_tiles(self):
        return -(-self.chip_rows // 64)

    @property
    def partial_shape(self):
        return (self.k_blocks, self.chip_rows)

    @property
    def scale_shape(self):
        return (self.k_blocks,)


def cooperative_block_geometry(types, result_type, attrs):
    values = (*types.values(), result_type)
    if any(logical_type(t).rank != 2 or any(not d.is_fixed for d in logical_type(t).shape) for t in values):
        return None
    distributed = tuple(t for t in values if isinstance(t, DistributedType))
    placement = distributed[0].placement if distributed else None
    if any(t.partial is not None or t.exclusive is not None or t.placement != placement for t in distributed):
        return None

    def policy(t, axis):
        return t.axis_policies[axis] if isinstance(t, DistributedType) else SBP.broadcast()

    value, weight, scale = (types[name] for name in ("value", "weight", "weight_scale"))
    v, w, s, result = map(logical_type, (value, weight, scale, result_type))
    weight_lanes = w.dtype.lane_count if isinstance(w.dtype, VectorType) else 1
    result_lanes = result.dtype.lane_count if isinstance(result.dtype, VectorType) else 1
    scalar_weight = w.dtype.elem_type if isinstance(w.dtype, VectorType) else w.dtype
    scalar_result = result.dtype.elem_type if isinstance(result.dtype, VectorType) else result.dtype
    if (
        v.dtype is not DType.BFLOAT16
        or scalar_weight is not DType.FLOAT8_E4M3FN
        or s.dtype is not DType.FLOAT32
        or scalar_result is not DType.BFLOAT16
        or v.shape[0].fixed_value != 1
        or result.shape[0].fixed_value != 1
        or attrs.get("activation_quantization") != "dynamic_fp8_sm90"
        or attrs.get("weight_dequant_dtype") is not None
        or attrs.get("weight_encoding") is not None
        or attrs.get("weight_block_k") != 128
        or v.shape[1].fixed_value % 128
        or w.shape[1].fixed_value * weight_lanes != v.shape[1].fixed_value
        or result.shape[1].fixed_value * result_lanes != w.shape[0].fixed_value
    ):
        return None
    row = policy(weight, 0)
    output_row = policy(result_type, 1)
    if isinstance(output_row, SBPSplit):
        output_row = scale_split_units(output_row, result_lanes, 1)
    if (
        output_row != row
        or row != SBP.broadcast()
        and not (
            isinstance(row, SBPSplit)
            and len(row.stages) == 1
            and isinstance(row.stages[0].distribution, ContiguousSplit)
        )
        or any(policy(value, i) != SBP.broadcast() for i in (0, 1))
        or policy(weight, 1) != SBP.broadcast()
        or policy(result_type, 0) != SBP.broadcast()
        or any(policy(scale, i) != SBP.broadcast() for i in (0, 1))
    ):
        return None
    local = local_tensor_type(weight) if isinstance(weight, DistributedType) else w
    hierarchy = tuple(placement.hierarchy) if placement else ()
    levels = placement.hierarchy_levels if placement else ""
    axes = tuple(i for i in getattr(row, "hierarchy_axes", ()) if levels[i] == "b")
    geometry = CooperativeBlockGeometry(local.shape[0].fixed_value, v.shape[1].fixed_value, hierarchy, levels, axes)
    return geometry if geometry.local_rows > 0 and geometry.reduction > 0 else None
