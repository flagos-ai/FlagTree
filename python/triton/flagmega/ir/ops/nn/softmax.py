# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Softmax on a materialized axis, accumulating in FP32."""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import DistributedType, SBP
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.axis import normalize_axis
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import DType


@op_definition("nn.softmax", namespace="nn", functional_name="softmax", display_name="NN.Softmax")
class Softmax(OpDefinition):
    const_evaluable = True
    value = input_parameter(is_tensor())
    axis = attribute_parameter(default=-1)

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        if isinstance(attrs["axis"], bool) or not isinstance(attrs["axis"], int):
            raise IRSchemaError("Softmax axis must be an integer.")
        return attrs

    @classmethod
    def infer_type(cls, inputs, attrs):
        source = cls.value.type_of(inputs)
        value = tensor_of(source)
        if value.dtype not in {DType.BFLOAT16, DType.FLOAT32}:
            raise IRSchemaError("Softmax requires scalar BF16/FP32 elements.")
        axis = normalize_axis(attrs["axis"], value.rank)
        if isinstance(source, DistributedType) and (source.partial is not None
                                                    or source.axis_policies[axis] != SBP.broadcast()):
            raise IRSchemaError("Softmax requires a materialized broadcast reduction axis.")
        return source

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        return value.float().softmax(dim=node.attrs["axis"]).to(value.dtype)

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_written=tensor_nbytes(node.type), notes=("softmax-fp32-reduction", ))
