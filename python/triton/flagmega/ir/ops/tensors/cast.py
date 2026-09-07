# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Tensor element type conversion."""

from typing import Mapping, Sequence

from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import DistributedType, IRType, Node, tensor_type
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import data_type, data_type_from_data, data_type_to_data


@op_definition("tensors.cast", namespace="tensors", functional_name="cast", display_name="Tensors.Cast")
class Cast(OpDefinition):
    const_evaluable = True
    value = input_parameter(is_tensor())
    dtype = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        dtype = attrs["dtype"]
        if isinstance(dtype, dict):
            dtype = data_type_from_data(dtype)
        return {"dtype": data_type_to_data(data_type(dtype))}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value = cls.value.type_of(inputs)
        tensor = tensor_of(value)
        output = tensor_type(
            data_type_from_data(attrs["dtype"]), tensor.shape, layout=tensor.layout
        )
        if not isinstance(value, DistributedType):
            return output
        return DistributedType(
            output, value.axis_policies, value.placement, partial=value.partial
        )

    @classmethod
    def evaluate(cls, node, arguments, context):
        return cls.value.read(arguments).to(
            dtype=context.torch_dtype(tensor_of(node.type).dtype)
        )

    @classmethod
    def python_attrs(cls, node: Node):
        # Public Python constructor accepts DataType, while stored attrs use its
        # deterministic serialization form.
        return {"dtype": data_type_from_data(node.attrs["dtype"])}

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(flops=0, bytes_read=None, bytes_written=tensor_nbytes(node.type), notes=("cast",))


__all__ = ["Cast"]
