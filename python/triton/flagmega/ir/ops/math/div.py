# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Elementwise division; broadcasting is explicit IR."""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import DistributedType
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import DType


@op_definition("math.div", namespace="math", functional_name="div", display_name="Math.Div")
class Div(OpDefinition):
    const_evaluable = True
    lhs = input_parameter(is_tensor())
    rhs = input_parameter(is_tensor())

    @classmethod
    def infer_type(cls, inputs, attrs):
        lhs, rhs = cls.lhs.type_of(inputs), cls.rhs.type_of(inputs)
        if lhs != rhs or tensor_of(lhs).dtype not in {DType.BFLOAT16, DType.FLOAT32}:
            raise IRSchemaError("Div requires identical BF16/FP32 tensor types; broadcast operands explicitly.")
        if isinstance(lhs, DistributedType) and lhs.partial is not None:
            raise IRSchemaError("Div requires materialized input.")
        return lhs

    @classmethod
    def evaluate(cls, node, arguments, context):
        return cls.lhs.read(arguments) / cls.rhs.read(arguments)

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_written=tensor_nbytes(node.type), notes=("elementwise-divide", ))
