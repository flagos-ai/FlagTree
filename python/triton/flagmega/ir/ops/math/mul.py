# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Math.Mul definition and its local behaviors."""

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import IRType, Node
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    input_parameter,
    op_definition,
    tensor_elements,
    tensor_nbytes,
)


@op_definition("math.mul", namespace="math", functional_name="mul", display_name="Math.Mul")
class Mul(OpDefinition):
    const_evaluable = True
    lhs = input_parameter(is_tensor())
    rhs = input_parameter(is_tensor())

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        if attrs:
            raise IRSchemaError("F.math.mul does not accept attributes.")
        lhs = cls.lhs.type_of(inputs)
        if cls.rhs.type_of(inputs) != lhs:
            raise IRSchemaError("F.math.mul requires identical tensor types.")
        return lhs

    @classmethod
    def evaluate(cls, node, arguments, context):
        return cls.lhs.read(arguments) * cls.rhs.read(arguments)

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        elements = tensor_elements(node.type) if hasattr(node.type, "shape") else None
        size = tensor_nbytes(node.type) if hasattr(node.type, "shape") else None
        return OpCost(flops=elements, bytes_read=None if size is None else size * 2, bytes_written=size)
