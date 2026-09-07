# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Math.Add definition and its local behaviors."""

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


@op_definition("math.add", namespace="math", functional_name="add", display_name="Math.Add")
class Add(OpDefinition):
    const_evaluable = True
    lhs = input_parameter(is_tensor())
    rhs = input_parameter(is_tensor())
    # Pointwise add reads each lhs element before writing the corresponding
    # result element, so an implementation may overwrite a dead lhs buffer.
    inplace_input_parameters = (lhs,)

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        if attrs:
            raise IRSchemaError("F.math.add does not accept attributes.")
        lhs = cls.lhs.type_of(inputs)
        if cls.rhs.type_of(inputs) != lhs:
            raise IRSchemaError("F.math.add requires identical tensor types.")
        return lhs

    @classmethod
    def evaluate(cls, node, arguments, context):
        return cls.lhs.read(arguments) + cls.rhs.read(arguments)

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        elements = tensor_elements(node.type) if hasattr(node.type, "shape") else None
        size = tensor_nbytes(node.type) if hasattr(node.type, "shape") else None
        return OpCost(flops=elements, bytes_read=None if size is None else size * 2, bytes_written=size)
