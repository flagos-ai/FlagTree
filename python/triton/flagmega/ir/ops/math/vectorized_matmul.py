# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Matrix multiplication over independently packed input axes."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import IRType, Node, TensorType, tensor_type
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.ops.tensors.pack import pack_physical
from triton.flagmega.ir.ops.tensors.unpack import unpack_physical
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import DType, VectorType
from triton.flagmega.ir.ops.math.matmul import matmul_value, normalize_output_data_type


@op_definition(
    "math.vectorized_matmul",
    namespace="math",
    functional_name="vectorized_matmul",
    display_name="Math.VectorizedMatMul",
)
class VectorizedMatMul(OpDefinition):
    const_evaluable = True
    lhs = input_parameter(is_tensor())
    rhs = input_parameter(is_tensor())
    lhs_axes = attribute_parameter()
    rhs_axes = attribute_parameter()
    output_axes = attribute_parameter()
    output_lanes = attribute_parameter()
    transpose_a = attribute_parameter(default=False)
    transpose_b = attribute_parameter(default=False)
    output_data_type = attribute_parameter(default=None)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        for name in ("lhs_axes", "rhs_axes", "output_axes", "output_lanes"):
            value = attrs[name]
            if not isinstance(value, (tuple, list)) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
                raise IRSchemaError(f"F.math.vectorized_matmul {name} must be an integer sequence.")
            attrs[name] = tuple(value)
        if len(attrs["output_axes"]) != len(attrs["output_lanes"]):
            raise IRSchemaError("VectorizedMatMul output axes and lanes must have equal length.")
        return normalize_output_data_type(attrs)

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        lhs = cls.lhs.type_of(inputs)
        rhs = cls.rhs.type_of(inputs)
        assert isinstance(lhs, TensorType) and isinstance(rhs, TensorType)
        lhs_logical = _logical_type(lhs, tuple(attrs["lhs_axes"]))
        rhs_logical = _logical_type(rhs, tuple(attrs["rhs_axes"]))
        lm, lk = (lhs_logical.shape[1], lhs_logical.shape[0]) if attrs["transpose_a"] else lhs_logical.shape
        rk, rn = (rhs_logical.shape[1], rhs_logical.shape[0]) if attrs["transpose_b"] else rhs_logical.shape
        if lk != rk or lhs_logical.dtype != rhs_logical.dtype:
            raise IRSchemaError("VectorizedMatMul logical input types are incompatible.")
        scalar_result = tensor_type(DType(attrs.get("output_data_type") or lhs_logical.dtype), (lm, rn), layout=lhs.layout)
        axes = tuple(int(value) for value in attrs["output_axes"])
        lanes = tuple(int(value) for value in attrs["output_lanes"])
        if not lanes:
            return scalar_result
        shape = list(scalar_result.shape)
        for axis, lane in zip(axes, lanes):
            shape[axis] = shape[axis] // lane
        return tensor_type(VectorType(scalar_result.dtype, lanes), shape, layout=scalar_result.layout)

    @classmethod
    def evaluate(cls, node, arguments, context):
        lhs = cls.lhs.read(arguments)
        rhs = cls.rhs.read(arguments)
        lhs_type = tensor_of(context.types[cls.lhs.read(node.inputs)])
        rhs_type = tensor_of(context.types[cls.rhs.read(node.inputs)])
        if isinstance(lhs_type.dtype, VectorType):
            lhs = unpack_physical(lhs, lhs_type.rank, lhs_type.dtype.lanes, tuple(node.attrs["lhs_axes"]))
        if isinstance(rhs_type.dtype, VectorType):
            rhs = unpack_physical(rhs, rhs_type.rank, rhs_type.dtype.lanes, tuple(node.attrs["rhs_axes"]))
        if node.attrs["transpose_a"]:
            lhs = lhs.transpose(-2, -1)
        if node.attrs["transpose_b"]:
            rhs = rhs.transpose(-2, -1)
        result = matmul_value(lhs, rhs, node.attrs.get("output_data_type"), context)
        lanes = tuple(node.attrs["output_lanes"])
        return result if not lanes else pack_physical(result, 2, lanes, tuple(node.attrs["output_axes"]))

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        assert isinstance(node.type, TensorType)
        return OpCost(flops=None, bytes_read=None, bytes_written=tensor_nbytes(node.type), notes=("vectorized-matmul",))


def _logical_type(value: TensorType, axes: tuple[int, ...]) -> TensorType:
    if not isinstance(value.dtype, VectorType):
        if axes:
            raise IRSchemaError("Scalar VectorizedMatMul input cannot declare vectorized axes.")
        return value
    if len(axes) != len(value.dtype.lanes):
        raise IRSchemaError("VectorizedMatMul axes must describe every input vector lane.")
    shape = list(value.shape)
    for axis, lane in zip(axes, value.dtype.lanes):
        shape[axis] = shape[axis] * lane
    return tensor_type(value.dtype.elem_type, shape, layout=value.layout)


__all__ = ["VectorizedMatMul"]
