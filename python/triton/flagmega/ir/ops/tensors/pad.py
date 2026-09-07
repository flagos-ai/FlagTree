# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""End-padding used to make vectorized axes lane-aligned."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import IRType, Node, TensorType, tensor_type
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.type_pattern import is_tensor


@op_definition("tensors.pad", namespace="tensors", functional_name="pad", display_name="Tensors.Pad")
class Pad(OpDefinition):
    const_evaluable = True
    numpy_materializable = True
    value = input_parameter(is_tensor())
    pad_end = attribute_parameter()
    pad_value = attribute_parameter(default=0.0)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        pads = attrs["pad_end"]
        if not isinstance(pads, (tuple, list)) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in pads
        ):
            raise IRSchemaError("F.tensors.pad pad_end must be a sequence of non-negative integers.")
        value = attrs["pad_value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise IRSchemaError("F.tensors.pad pad_value must be numeric.")
        return {"pad_end": tuple(pads), "pad_value": value}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value_type = cls.value.type_of(inputs)
        assert isinstance(value_type, TensorType)
        pads = tuple(int(value) for value in attrs["pad_end"])
        if len(pads) != value_type.rank:
            raise IRSchemaError(f"F.tensors.pad expected {value_type.rank} pad values, got {len(pads)}.")
        return tensor_type(
            value_type.dtype,
            tuple(dimension + pad for dimension, pad in zip(value_type.shape, pads)),
            layout=value_type.layout,
        )

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        value_type = tensor_of(context.types[cls.value.read(node.inputs)])
        pads = tuple(int(item) for item in node.attrs["pad_end"])
        output_shape = list(value.shape)
        for axis, pad in enumerate(pads):
            output_shape[axis] += pad
        output = value.new_full(tuple(output_shape), node.attrs["pad_value"])
        slices = tuple(slice(0, int(value.shape[axis])) for axis in range(value_type.rank))
        output[slices] = value
        return output

    @classmethod
    def materialize_numpy(cls, node, arguments, context):
        value = cls.value.read(arguments)
        value_type = tensor_of(context.types[cls.value.read(node.inputs)])
        output = context.full(node.type, node.attrs["pad_value"])
        slices = tuple(
            slice(0, int(value.shape[axis])) for axis in range(value_type.rank)
        )
        output[slices] = value
        return output

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        size = tensor_nbytes(node.type) if isinstance(node.type, TensorType) else None
        return OpCost(bytes_read=size, bytes_written=size, notes=("vector-pad",))


__all__ = ["Pad"]
