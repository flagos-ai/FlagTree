# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Elementwise exponential with FP32 computation and a declared output boundary."""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import DistributedType
from triton.flagmega.ir.ops.core import OpCost, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.ops.pointwise import SameTypePointwiseOp
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import DType


@op_definition("math.exp", namespace="math", functional_name="exp", display_name="Math.Exp")
class Exp(SameTypePointwiseOp):
    const_evaluable = True
    value = input_parameter(is_tensor())

    @classmethod
    def infer_type(cls, inputs, attrs):
        value = cls.value.type_of(inputs)
        if tensor_of(value).dtype not in {DType.BFLOAT16, DType.FLOAT32}:
            raise IRSchemaError("Exp requires scalar BF16/FP32 elements.")
        if isinstance(value, DistributedType) and value.partial is not None:
            raise IRSchemaError("Exp requires materialized input.")
        return value

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        return value.float().exp().to(value.dtype)

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_written=tensor_nbytes(node.type), notes=("exp", ))
