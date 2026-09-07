# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Pack logical tensor axes into an nncase-compatible ``VectorType``."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import (
    SBPSplit,
    local_shape,
    scale_split_units,
)
from triton.flagmega.ir.dim_expr import try_div_exactly
from triton.flagmega.ir.model import DistributedType, IRType, Node, TensorType, tensor_type
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import DType, VectorType


@op_definition("tensors.pack", namespace="tensors", functional_name="pack", display_name="Tensors.Pack")
class Pack(OpDefinition):
    const_evaluable = True
    numpy_materializable = True
    """Move one or more logical lane factors into the element data type.

    ``lanes[i]`` belongs to ``axes[i]``. Axes may repeat, matching nncase's
    axis-lane-product semantics. Packing an existing VectorType prepends the
    new lanes, which is needed by vectorization propagation.
    """

    value = input_parameter(is_tensor())
    lanes = attribute_parameter()
    axes = attribute_parameter(default=None)
    axis = attribute_parameter(default=None)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        lanes = _positive_ints(attrs["lanes"], "lanes")
        axes_value = attrs["axes"]
        axis_value = attrs["axis"]
        if axes_value is not None and axis_value is not None:
            raise IRSchemaError("F.tensors.pack accepts either axes or axis, not both.")
        if axes_value is None:
            if axis_value is None:
                axis_value = -1
            if isinstance(axis_value, bool) or not isinstance(axis_value, int):
                raise IRSchemaError("F.tensors.pack axis must be an integer.")
            return {"lanes": lanes, "axis": axis_value}
        else:
            axes = _ints(axes_value, "axes")
        if len(lanes) != len(axes):
            raise IRSchemaError("F.tensors.pack lanes and axes must have the same length.")
        return {"lanes": lanes, "axes": axes}

    @classmethod
    def ir_attrs(cls, attrs: Mapping[str, object]) -> Mapping[str, object]:
        return attrs

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        source_type = cls.value.type_of(inputs)
        value_type = tensor_of(source_type)
        element_type = value_type.dtype.elem_type if isinstance(value_type.dtype, VectorType) else value_type.dtype
        if element_type == DType.BOOL:
            raise IRSchemaError("F.tensors.pack does not support BooleanType; use a target mask-vector op.")
        lanes = tuple(int(value) for value in attrs["lanes"])
        axes = _pack_axes(attrs, len(lanes), value_type.rank)
        shape = list(value_type.shape)
        for axis, lane_product in axis_lane_products(lanes, axes).items():
            packed_extent = try_div_exactly(shape[axis], lane_product)
            if packed_extent is None:
                raise IRSchemaError(
                    f"F.tensors.pack axis {axis} extent {shape[axis]} is not provably divisible "
                    f"by {lane_product}; pad it before packing.")
            shape[axis] = packed_extent
        if isinstance(value_type.dtype, VectorType):
            dtype = VectorType(value_type.dtype.elem_type, (*lanes, *value_type.dtype.lanes))
        else:
            dtype = VectorType(value_type.dtype, lanes)
        packed = tensor_type(dtype, shape, layout=value_type.layout)
        if not isinstance(source_type, DistributedType):
            return packed
        policies = list(source_type.axis_policies)
        local = local_shape(source_type)
        for axis, lane_product in axis_lane_products(lanes, axes).items():
            policy = policies[axis]
            if not isinstance(policy, SBPSplit):
                continue
            if try_div_exactly(local[axis], lane_product) is None:
                raise IRSchemaError(
                    f"F.tensors.pack axis {axis} split boundary cuts lane group "
                    f"{lane_product}."
                )
            scaled = scale_split_units(policy, 1, lane_product)
            if scaled is None:
                raise IRSchemaError(
                    f"F.tensors.pack cannot scale axis {axis} split policy by "
                    f"lane group {lane_product}."
                )
            policies[axis] = scaled
        return DistributedType(
            packed,
            tuple(policies),
            source_type.placement,
            source_type.partial,
        )

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        input_type = tensor_of(context.types[cls.value.read(node.inputs)])
        lanes = tuple(int(item) for item in node.attrs["lanes"])
        axes = _pack_axes(node.attrs, len(lanes), input_type.rank)
        old_lanes = input_type.dtype.lanes if isinstance(input_type.dtype, VectorType) else ()
        return pack_physical(value, input_type.rank, lanes, axes, old_lanes)

    @classmethod
    def materialize_numpy(cls, node, arguments, context):
        value = cls.value.read(arguments)
        input_type = tensor_of(context.types[cls.value.read(node.inputs)])
        lanes = tuple(int(item) for item in node.attrs["lanes"])
        axes = _pack_axes(node.attrs, len(lanes), input_type.rank)
        old_lanes = input_type.dtype.lanes if isinstance(input_type.dtype, VectorType) else ()
        return context.pack(value, input_type.rank, lanes, axes, old_lanes)

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        size = tensor_nbytes(node.type) if isinstance(node.type, TensorType) else None
        return OpCost(bytes_read=size, bytes_written=size, notes=("vector-pack",))


def pack_physical(value, outer_rank: int, lanes: tuple[int, ...], axes: tuple[int, ...], old_lanes=()):
    """Reference physical reshape used by Pack and vectorized evaluators."""

    normalized = normalize_axes(axes, outer_rank)
    by_axis: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for lane_index, (axis, lane) in enumerate(zip(normalized, lanes)):
        by_axis[axis].append((lane_index, lane))

    split_shape: list[int] = []
    outer_positions: list[int] = []
    lane_positions: dict[int, int] = {}
    cursor = 0
    for axis in range(outer_rank):
        factors = by_axis.get(axis, ())
        product = _product(lane for _, lane in factors)
        extent = int(value.shape[axis])
        if extent % product:
            raise IRSchemaError(f"Pack physical axis {axis} extent {extent} is not divisible by {product}.")
        split_shape.append(extent // product)
        outer_positions.append(cursor)
        cursor += 1
        for lane_index, lane in factors:
            split_shape.append(lane)
            lane_positions[lane_index] = cursor
            cursor += 1
    old_positions = tuple(range(cursor, cursor + len(old_lanes)))
    split_shape.extend(int(lane) for lane in old_lanes)
    split = value.reshape(tuple(split_shape))
    permutation = (*outer_positions, *(lane_positions[index] for index in range(len(lanes))), *old_positions)
    return split.permute(*permutation).contiguous()


def normalize_axes(axes: Sequence[int], rank: int) -> tuple[int, ...]:
    normalized: list[int] = []
    for axis in axes:
        value = axis + rank if axis < 0 else axis
        if value < 0 or value >= rank:
            raise IRSchemaError(f"Tensor axis {axis} is out of range for rank {rank}.")
        normalized.append(value)
    return tuple(normalized)


def axis_lane_products(lanes: Sequence[int], axes: Sequence[int]) -> dict[int, int]:
    products: dict[int, int] = {}
    for lane, axis in zip(lanes, axes):
        products[axis] = products.get(axis, 1) * lane
    return products


def _pack_axes(attrs: Mapping[str, object], lane_count: int, rank: int) -> tuple[int, ...]:
    if "axes" in attrs:
        values = tuple(int(value) for value in attrs["axes"])
    else:
        values = (int(attrs["axis"]),) * lane_count
    return normalize_axes(values, rank)


def _ints(value: object, name: str) -> tuple[int, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        return (value,)
    if not isinstance(value, (tuple, list)) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise IRSchemaError(f"F.tensors.pack {name} must be an integer sequence.")
    return tuple(value)


def _positive_ints(value: object, name: str) -> tuple[int, ...]:
    values = _ints(value, name)
    if not values or any(item <= 0 for item in values):
        raise IRSchemaError(f"F.tensors.pack {name} must contain positive integers.")
    return values


def _product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


__all__ = ["Pack", "axis_lane_products", "normalize_axes", "pack_physical"]
