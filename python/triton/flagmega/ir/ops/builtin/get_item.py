# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Tuple GetItem definition and its local behaviors."""

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import IRType, Node
from triton.flagmega.ir.ops.core import (
    NodeRef,
    OpCost,
    OpDefinition,
    PythonCall,
    attribute_parameter,
    input_parameter,
    op_definition,
)
from triton.flagmega.ir.type_pattern import is_tuple


@op_definition(
    "builtin.get_item",
    namespace="tensors",
    functional_name="get_item",
    display_name="GetItem",
)
class GetItem(OpDefinition):
    const_evaluable = True
    # Tuple projection must return the selected field's type verbatim. The
    # broadcast-lifting fallback re-infers from logical inputs and would lift a
    # scalar field of an otherwise distributed tuple into a broadcast
    # DistributedType, corrupting checkpoint rebuilds at distributed stages.
    supports_broadcast_lifting = False
    value = input_parameter(is_tuple())
    index = attribute_parameter(positional=True)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        if set(attributes) != {"index"}:
            raise IRSchemaError("F.tensors.get_item requires exactly one index attribute.")
        index = attributes["index"]
        if isinstance(index, bool) or not isinstance(index, int):
            raise IRSchemaError("F.tensors.get_item index must be an integer.")
        return {"index": index}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value_type = cls.value.type_of(inputs)
        index = int(cls.index.read(inputs, attrs))
        if index < 0 or index >= len(value_type.fields):
            raise IRSchemaError(f"Tuple index {index!r} is out of range.")
        return value_type.fields[index]

    @classmethod
    def evaluate(cls, node, arguments, context):
        return cls.value.read(arguments)[int(cls.index.read(arguments, node.attrs))]

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(notes=("tuple-projection",))

    @classmethod
    def python_call(cls, node: Node) -> PythonCall:
        keywords = {"name": node.id}
        if node.metadata:
            keywords["metadata"] = node.metadata
        return PythonCall(
            "F.tensors.get_item",
            (NodeRef(cls.value.read(node.inputs)), int(cls.index.read(node.inputs, node.attrs))),
            keywords,
        )
