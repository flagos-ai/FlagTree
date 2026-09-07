# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Slice padded outer dimensions back to an exact static logical shape."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import IRType, Node, TensorType, tensor_type
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.type_pattern import is_tensor


@op_definition(
    "tensors.slice_to_shape",
    namespace="tensors",
    functional_name="slice_to_shape",
    display_name="Tensors.SliceToShape",
)
class SliceToShape(OpDefinition):
    const_evaluable = True
    numpy_materializable = True
    value = input_parameter(is_tensor())
    shape = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        shape = attrs["shape"]
        if not isinstance(shape, (tuple, list)) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape
        ):
            raise IRSchemaError("F.tensors.slice_to_shape shape must be a sequence of non-negative integers.")
        return {"shape": tuple(shape)}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value_type = cls.value.type_of(inputs)
        assert isinstance(value_type, TensorType)
        shape = tuple(int(value) for value in attrs["shape"])
        if len(shape) != value_type.rank:
            raise IRSchemaError(f"F.tensors.slice_to_shape expected rank {value_type.rank}, got {len(shape)}.")
        for requested, extent in zip(shape, value_type.shape):
            if extent.value is not None and requested > extent.fixed_value:
                raise IRSchemaError("F.tensors.slice_to_shape cannot grow a tensor dimension.")
        return tensor_type(value_type.dtype, shape, layout=value_type.layout)

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        value_type = tensor_of(context.types[cls.value.read(node.inputs)])
        shape = tuple(int(item) for item in node.attrs["shape"])
        slices = tuple(slice(0, extent) for extent in shape)
        slices += tuple(slice(None) for _ in range(value.ndim - value_type.rank))
        return value[slices].contiguous()

    @classmethod
    def materialize_numpy(cls, node, arguments, context):
        value = cls.value.read(arguments)
        value_type = tensor_of(context.types[cls.value.read(node.inputs)])
        shape = tuple(int(item) for item in node.attrs["shape"])
        slices = tuple(slice(0, extent) for extent in shape)
        slices += tuple(slice(None) for _ in range(value.ndim - value_type.rank))
        return context.as_contiguous(value[slices])

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        size = tensor_nbytes(node.type) if isinstance(node.type, TensorType) else None
        return OpCost(bytes_read=size, bytes_written=size, notes=("vector-unpad",))


__all__ = ["SliceToShape"]
