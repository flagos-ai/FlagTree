# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""builtin.var definition and its local behaviors."""

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.model import IRModule, Node
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, op_definition, tensor_nbytes


@op_definition("builtin.var", display_name="Var")
class Var(OpDefinition):
    name = attribute_parameter()

    @classmethod
    def verify(cls, node: Node, module: IRModule) -> None:
        cls.verify_arity(node)
        if set(node.attrs) != {"name"} or not str(node.attrs["name"]):
            raise IRVerificationError("builtin.var requires one non-empty name attribute.", node_id=node.id)

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = context.input_value(node)
        context.validate_value(node, value)
        return value

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        size = tensor_nbytes(node.type) if hasattr(node.type, "dtype") else None
        return OpCost(bytes_read=size, notes=("caller-input",))
