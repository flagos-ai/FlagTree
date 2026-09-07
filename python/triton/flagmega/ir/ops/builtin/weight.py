# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""builtin.weight definition and its local behaviors."""

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.model import IRModule, Node
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, op_definition, tensor_nbytes


@op_definition("builtin.weight", display_name="WeightRef")
class Weight(OpDefinition):
    constant_source = True
    numpy_materializable = True
    name = attribute_parameter()
    source = attribute_parameter()
    key = attribute_parameter()
    source_hash = attribute_parameter(default=None)

    @classmethod
    def verify(cls, node: Node, module: IRModule) -> None:
        cls.verify_arity(node)
        required = {"name", "source", "key"}
        if not required.issubset(node.attrs) or any(not str(node.attrs[key]) for key in required):
            raise IRVerificationError("builtin.weight requires non-empty name/source/key attributes.", node_id=node.id)

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = context.weight_value(node)
        context.validate_value(node, value)
        return value

    @classmethod
    def materialize_numpy(cls, node, arguments, context):
        del arguments
        return context.weight_value(node)

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        size = tensor_nbytes(node.type) if hasattr(node.type, "dtype") else None
        return OpCost(bytes_read=size, notes=("readonly-data",))
