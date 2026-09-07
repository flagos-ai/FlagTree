# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Unpack leading VectorType lanes back into logical tensor axes."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import SBPSplit, scale_split_units
from triton.flagmega.ir.model import DistributedType, IRType, Node, TensorType, tensor_type
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.ops.tensors.pack import normalize_axes
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import VectorType


@op_definition("tensors.unpack", namespace="tensors", functional_name="unpack", display_name="Tensors.Unpack")
class Unpack(OpDefinition):
    const_evaluable = True
    numpy_materializable = True
    value = input_parameter(is_tensor())
    axes = attribute_parameter(default=None)
    axis = attribute_parameter(default=None)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        axes_value = attrs["axes"]
        axis_value = attrs["axis"]
        if axes_value is not None and axis_value is not None:
            raise IRSchemaError("F.tensors.unpack accepts either axes or axis, not both.")
        if axes_value is None:
            if axis_value is None:
                axis_value = -1
            if isinstance(axis_value, bool) or not isinstance(axis_value, int):
                raise IRSchemaError("F.tensors.unpack axis must be an integer.")
            return {"axis": axis_value}
        elif isinstance(axes_value, int) and not isinstance(axes_value, bool):
            axes = (axes_value,)
        elif isinstance(axes_value, (tuple, list)) and all(
            isinstance(value, int) and not isinstance(value, bool) for value in axes_value
        ):
            axes = tuple(axes_value)
        else:
            raise IRSchemaError("F.tensors.unpack axes must be an integer sequence.")
        if not axes:
            raise IRSchemaError("F.tensors.unpack axes cannot be empty.")
        return {"axes": axes}

    @classmethod
    def ir_attrs(cls, attrs: Mapping[str, object]) -> Mapping[str, object]:
        return attrs

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        source_type = cls.value.type_of(inputs)
        value_type = tensor_of(source_type)
        if not isinstance(value_type.dtype, VectorType):
            raise IRSchemaError("F.tensors.unpack expects a VectorType tensor.")
        axes = _unpack_axes(attrs, value_type)
        if len(axes) > len(value_type.dtype.lanes):
            raise IRSchemaError("F.tensors.unpack cannot consume more axes than the VectorType has lanes.")
        consumed = value_type.dtype.lanes[:len(axes)]
        shape = list(value_type.shape)
        for axis, lane in zip(axes, consumed):
            shape[axis] = shape[axis] * lane
        remaining = value_type.dtype.lanes[len(axes):]
        dtype = value_type.dtype.elem_type if not remaining else VectorType(value_type.dtype.elem_type, remaining)
        unpacked = tensor_type(dtype, shape, layout=value_type.layout)
        if not isinstance(source_type, DistributedType):
            return unpacked
        policies = list(source_type.axis_policies)
        products: dict[int, int] = {}
        for axis, lane in zip(axes, consumed):
            products[axis] = products.get(axis, 1) * lane
        for axis, lane_product in products.items():
            policy = policies[axis]
            if not isinstance(policy, SBPSplit):
                continue
            scaled = scale_split_units(policy, lane_product, 1)
            if scaled is None:  # pragma: no cover - multiplying positive units is exact.
                raise IRSchemaError(
                    f"F.tensors.unpack cannot scale axis {axis} split policy."
                )
            policies[axis] = scaled
        return DistributedType(
            unpacked,
            tuple(policies),
            source_type.placement,
            source_type.partial,
        )

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        input_type = tensor_of(context.types[cls.value.read(node.inputs)])
        assert isinstance(input_type.dtype, VectorType)
        axes = _unpack_axes(node.attrs, input_type)
        return unpack_physical(value, input_type.rank, input_type.dtype.lanes, axes)

    @classmethod
    def materialize_numpy(cls, node, arguments, context):
        value = cls.value.read(arguments)
        input_type = tensor_of(context.types[cls.value.read(node.inputs)])
        assert isinstance(input_type.dtype, VectorType)
        axes = _unpack_axes(node.attrs, input_type)
        normalized = normalize_axes(axes, input_type.rank)
        consumed = input_type.dtype.lanes[:len(normalized)]
        remaining_count = len(input_type.dtype.lanes) - len(consumed)
        by_axis = defaultdict(list)
        for lane_index, (axis, lane) in enumerate(zip(normalized, consumed)):
            by_axis[axis].append((lane_index, lane))
        interleaved = []
        for axis in range(input_type.rank):
            interleaved.append(axis)
            interleaved.extend(
                input_type.rank + lane_index
                for lane_index, _ in by_axis.get(axis, ())
            )
        interleaved.extend(range(
            input_type.rank + len(consumed),
            input_type.rank + len(consumed) + remaining_count,
        ))
        split = context.as_contiguous(value.transpose(tuple(interleaved)))
        result_shape = []
        for axis in range(input_type.rank):
            product = 1
            for _, lane in by_axis.get(axis, ()):
                product *= lane
            result_shape.append(int(value.shape[axis]) * product)
        result_shape.extend(int(lane) for lane in input_type.dtype.lanes[len(consumed):])
        return split.reshape(tuple(result_shape))

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        size = tensor_nbytes(node.type) if isinstance(node.type, TensorType) else None
        return OpCost(bytes_read=size, bytes_written=size, notes=("vector-unpack",))


def unpack_physical(value, outer_rank: int, lanes: tuple[int, ...], axes: tuple[int, ...]):
    """Reference physical reshape for a leading subset of vector lanes."""

    normalized = normalize_axes(axes, outer_rank)
    consumed = lanes[:len(normalized)]
    remaining_count = len(lanes) - len(consumed)
    by_axis: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for lane_index, (axis, lane) in enumerate(zip(normalized, consumed)):
        by_axis[axis].append((lane_index, lane))

    interleaved: list[int] = []
    for axis in range(outer_rank):
        interleaved.append(axis)
        interleaved.extend(outer_rank + lane_index for lane_index, _ in by_axis.get(axis, ()))
    interleaved.extend(range(outer_rank + len(consumed), outer_rank + len(consumed) + remaining_count))
    split = value.permute(*interleaved).contiguous()
    result_shape: list[int] = []
    for axis in range(outer_rank):
        product = 1
        for _, lane in by_axis.get(axis, ()):
            product *= lane
        result_shape.append(int(value.shape[axis]) * product)
    result_shape.extend(int(lane) for lane in lanes[len(consumed):])
    return split.reshape(tuple(result_shape))


def _unpack_axes(attrs: Mapping[str, object], value_type: TensorType) -> tuple[int, ...]:
    assert isinstance(value_type.dtype, VectorType)
    if "axes" in attrs:
        values = tuple(int(value) for value in attrs["axes"])
    else:
        values = (int(attrs["axis"]),) * len(value_type.dtype.lanes)
    return normalize_axes(values, value_type.rank)


__all__ = ["Unpack", "unpack_physical"]
