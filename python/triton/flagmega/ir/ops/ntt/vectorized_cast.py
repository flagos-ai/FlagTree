# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Cast between vector element layouts without changing scalar values."""

from __future__ import annotations

from typing import Mapping, Sequence
from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import SBPSplit, scale_split_units
from triton.flagmega.ir.model import DistributedType, IRType, Node, TensorType, tensor_type
from triton.flagmega.ir.dim_expr import try_div_exactly
from triton.flagmega.ir.ops.core import (
    CostKind,
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_elements,
    tensor_nbytes,
)
from triton.flagmega.ir.ops.tensors.pack import axis_lane_products, normalize_axes, pack_physical
from triton.flagmega.ir.ops.tensors.unpack import unpack_physical
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import VectorType, data_type, data_type_from_data


@op_definition(
    "ntt.vectorized_cast",
    namespace="ntt",
    functional_name="vectorized_cast",
    display_name="NTT.VectorizedCast",
)
class VectorizedCast(OpDefinition):
    """Convert vector elements and repack lane groups on logical axes.

    This is the Python counterpart of nncase ``IR.NTT.VectorizedCast``.  The
    input and output vector types may have different lane sizes so long as
    unpacking the input and repacking the output preserves the same scalar
    logical shape.
    """

    const_evaluable = True
    value = input_parameter(is_tensor())
    new_type = attribute_parameter()
    vectorize_axes = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        new_type = data_type(attrs["new_type"])
        if not isinstance(new_type, VectorType):
            raise IRSchemaError("F.ntt.vectorized_cast new_type must be a VectorType.")
        axes_value = attrs["vectorize_axes"]
        if isinstance(axes_value, int) and not isinstance(axes_value, bool):
            axes = (axes_value,)
        elif isinstance(axes_value, (tuple, list)) and all(
            isinstance(value, int) and not isinstance(value, bool) for value in axes_value
        ):
            axes = tuple(axes_value)
        else:
            raise IRSchemaError("F.ntt.vectorized_cast vectorize_axes must be an integer sequence.")
        if not axes:
            raise IRSchemaError("F.ntt.vectorized_cast vectorize_axes cannot be empty.")
        return {
            "new_type": {
                "kind": "vector",
                "elem_type": new_type.elem_type.value,
                "lanes": tuple(new_type.lanes),
            },
            "vectorize_axes": axes,
        }

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        source_type = cls.value.type_of(inputs)
        input_type = tensor_of(source_type)
        if not isinstance(input_type.dtype, VectorType):
            raise IRSchemaError("F.ntt.vectorized_cast requires a VectorType input tensor.")
        output_dtype = data_type_from_data(attrs["new_type"])
        assert isinstance(output_dtype, VectorType)
        input_axes, output_axes = cast_vector_axes(input_type.dtype.lanes, output_dtype.lanes,
                                                 attrs["vectorize_axes"], input_type.rank)
        shape = list(input_type.shape)
        input_products = axis_lane_products(input_type.dtype.lanes, input_axes)
        output_products = axis_lane_products(output_dtype.lanes, output_axes)
        for axis, input_lane in input_products.items():
            output_lane = output_products[axis]
            scalar_extent = shape[axis] * input_lane
            packed_extent = try_div_exactly(scalar_extent, output_lane)
            if packed_extent is None:
                raise IRSchemaError(
                    f"F.ntt.vectorized_cast unpacked axis {axis} extent {scalar_extent} "
                    f"is not provably divisible by output lane {output_lane}."
                )
            shape[axis] = packed_extent
        output = tensor_type(output_dtype, shape, layout=input_type.layout)
        if not isinstance(source_type, DistributedType):
            return output
        policies = list(source_type.axis_policies)
        for axis, input_lane in input_products.items():
            output_lane = output_products[axis]
            policy = policies[axis]
            if not isinstance(policy, SBPSplit):
                continue
            scaled = scale_split_units(policy, input_lane, output_lane)
            if scaled is None:
                raise IRSchemaError(
                    f"F.ntt.vectorized_cast cannot scale axis {axis} split policy "
                    f"from lane {input_lane} to lane {output_lane}."
                )
            policies[axis] = scaled
        return DistributedType(
            output,
            tuple(policies),
            source_type.placement,
            partial=source_type.partial,
        )

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        input_type = tensor_of(context.types[cls.value.read(node.inputs)])
        assert isinstance(input_type.dtype, VectorType)
        output_type = tensor_of(node.type)
        assert isinstance(output_type.dtype, VectorType)
        input_axes, output_axes = cast_vector_axes(input_type.dtype.lanes, output_type.dtype.lanes,
                                                 node.attrs["vectorize_axes"], input_type.rank)
        scalar = unpack_physical(value, input_type.rank, input_type.dtype.lanes, input_axes)
        converted = scalar.to(dtype=context.torch_dtype(output_type.dtype.elem_type))
        return pack_physical(converted, output_type.rank, output_type.dtype.lanes, output_axes)

    @classmethod
    def python_attrs(cls, node: Node):
        return {
            "new_type": data_type_from_data(node.attrs["new_type"]),
            "vectorize_axes": tuple(int(value) for value in node.attrs["vectorize_axes"]),
        }

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        elements = tensor_elements(node.type)
        return OpCost(
            flops=elements,
            bytes_read=None,
            bytes_written=tensor_nbytes(node.type),
            kind=CostKind.ANALYTIC,
            model="flagmega.ntt-vectorized-cast/v1",
            notes=("vectorized-cast",),
        )


def cast_vector_axes(input_lanes, output_lanes, axes, rank):
    axes = normalize_axes(tuple(int(axis) for axis in axes), rank)
    if axes and len(set(axes)) == 1:
        return (axes[0],) * len(input_lanes), (axes[0],) * len(output_lanes)
    if len(input_lanes) != len(output_lanes):
        short, long = sorted((tuple(input_lanes), tuple(output_lanes)), key=len)
        prefix = len(short) - 1
        if short[:-1] == long[:prefix] and short[-1] == prod(long[prefix:]):
            if len(axes) == len(short):
                short_axes, long_axes = axes, (*axes[:prefix], *((axes[-1],) * (len(long) - prefix)))
            elif len(axes) == len(long) and len(set(axes[prefix:])) == 1:
                short_axes, long_axes = (*axes[:prefix], axes[-1]), axes
            else:
                raise IRSchemaError("VectorizedCast packet splitting must preserve logical axes.")
            return (short_axes, long_axes) if len(input_lanes) < len(output_lanes) else (long_axes, short_axes)
    if len(axes) != len(input_lanes) or len(axes) != len(output_lanes):
        raise IRSchemaError("VectorizedCast needs one input/output lane per axis, or one shared logical axis.")
    return axes, axes


__all__ = ["VectorizedCast"]
