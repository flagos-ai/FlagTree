# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""RMSNorm over a packed reduction axis."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import IRType, Node, TensorType
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.ops.nn.rms_norm import rms_norm
from triton.flagmega.ir.ops.tensors.pack import pack_physical
from triton.flagmega.ir.ops.tensors.unpack import unpack_physical
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import VectorType


@op_definition(
    "nn.vectorized_rms_norm",
    namespace="nn",
    functional_name="vectorized_rms_norm",
    display_name="NN.VectorizedRMSNorm",
)
class VectorizedRMSNorm(OpDefinition):
    value = input_parameter(is_tensor())
    weight = input_parameter(is_tensor())
    value_axes = attribute_parameter()
    weight_axes = attribute_parameter()
    logical_extent = attribute_parameter()
    epsilon = attribute_parameter()
    weight_bias = attribute_parameter(default=1.0)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        attrs["value_axes"] = tuple(int(value) for value in attrs["value_axes"])
        attrs["weight_axes"] = tuple(int(value) for value in attrs["weight_axes"])
        attrs["logical_extent"] = int(attrs["logical_extent"])
        attrs["epsilon"] = float(attrs["epsilon"])
        attrs["weight_bias"] = float(attrs["weight_bias"])
        if attrs["logical_extent"] <= 0 or attrs["epsilon"] <= 0:
            raise IRSchemaError("VectorizedRMSNorm requires positive logical_extent and epsilon.")
        return attrs

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value = cls.value.type_of(inputs)
        weight = cls.weight.type_of(inputs)
        assert isinstance(value, TensorType) and isinstance(weight, TensorType)
        if (
            not isinstance(value.dtype, VectorType) or not isinstance(weight.dtype, VectorType)
            or value.dtype != weight.dtype or value.dtype.elem_type != weight.dtype.elem_type
            or value.rank < 1 or weight.rank != 1
        ):
            raise IRSchemaError("VectorizedRMSNorm requires compatible packed value and weight tensors.")
        return value

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        weight = cls.weight.read(arguments)
        value_type = tensor_of(context.types[cls.value.read(node.inputs)])
        weight_type = tensor_of(context.types[cls.weight.read(node.inputs)])
        assert isinstance(value_type.dtype, VectorType)
        assert isinstance(weight_type.dtype, VectorType)
        value_axes = tuple(node.attrs["value_axes"])
        weight_axes = tuple(node.attrs["weight_axes"])
        scalar_value = unpack_physical(value, value_type.rank, value_type.dtype.lanes, value_axes)
        scalar_weight = unpack_physical(weight, weight_type.rank, weight_type.dtype.lanes, weight_axes)
        extent = int(node.attrs["logical_extent"])
        normalized = rms_norm(
            scalar_value[..., :extent],
            scalar_weight[:extent],
            epsilon=float(node.attrs["epsilon"]),
            weight_bias=float(node.attrs["weight_bias"]),
        )
        padded = scalar_value.new_zeros(scalar_value.shape)
        padded[..., :extent] = normalized
        return pack_physical(padded, value_type.rank, value_type.dtype.lanes, value_axes)

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        assert isinstance(node.type, TensorType)
        return OpCost(flops=None, bytes_read=None, bytes_written=tensor_nbytes(node.type), notes=("vectorized-rms-norm",))


__all__ = ["VectorizedRMSNorm"]
