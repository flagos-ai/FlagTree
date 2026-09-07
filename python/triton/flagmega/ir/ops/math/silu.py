# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Math.Silu definition and its local behaviors."""

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


@op_definition("math.silu", namespace="math", functional_name="silu", display_name="Math.Silu")
class Silu(OpDefinition):
    const_evaluable = True
    value = input_parameter(is_tensor())

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        if attrs:
            raise IRSchemaError("F.math.silu does not accept attributes.")
        return cls.value.type_of(inputs)

    @classmethod
    def evaluate(cls, node, arguments, context):
        return context.torch.nn.functional.silu(cls.value.read(arguments))

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        elements = tensor_elements(node.type) if hasattr(node.type, "shape") else None
        size = tensor_nbytes(node.type) if hasattr(node.type, "shape") else None
        return OpCost(flops=None if elements is None else elements * 4, bytes_read=size, bytes_written=size)
