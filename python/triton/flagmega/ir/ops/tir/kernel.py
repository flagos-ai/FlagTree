# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Semantic TIR kernel call definition and its local behaviors."""

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.ir.model import Effect, IRModule, IRType, Node, PURE
from triton.flagmega.ir.type_pattern import is_ir_type
from triton.flagmega.ir.ops.core import (
    NodeRef,
    OpCost,
    OpDefinition,
    PythonCall,
    attribute_parameter,
    op_definition,
    variadic_input_parameter,
)


@op_definition("tir.kernel", namespace="tir", functional_name="kernel", display_name="T.Kernel")
class Kernel(OpDefinition):
    supports_broadcast_lifting = False
    arguments = variadic_input_parameter(is_ir_type())
    result_type = attribute_parameter()
    semantic_op = attribute_parameter()
    candidate = attribute_parameter()
    kernel_parameters = attribute_parameter("parameters")
    facts = attribute_parameter()
    semantic_attrs = attribute_parameter()
    kernel_effect = attribute_parameter("effect", default=PURE)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        required = {"result_type", "semantic_op", "candidate", "parameters", "facts", "semantic_attrs"}
        allowed = required | {"effect"}
        if not required.issubset(attributes) or not set(attributes).issubset(allowed):
            raise IRSchemaError(f"F.tir.kernel requires attributes {sorted(required)} and optional effect.")
        if not isinstance(attributes["result_type"], IRType):
            raise IRSchemaError("F.tir.kernel result_type must be an IRType.")
        effect = attributes.get("effect", PURE)
        if not isinstance(effect, Effect):
            raise IRSchemaError("F.tir.kernel effect must be an Effect.")
        return {
            "result_type": attributes["result_type"],
            "semantic_op": str(attributes["semantic_op"]),
            "candidate": str(attributes["candidate"]),
            "parameters": dict(attributes["parameters"]),
            "facts": dict(attributes["facts"]),
            "semantic_attrs": dict(attributes["semantic_attrs"]),
            "effect": effect,
        }

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        return cls.result_type.read(inputs, attrs)

    @classmethod
    def infer_effect(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> Effect:
        return cls.kernel_effect.read(inputs, attrs)

    @classmethod
    def ir_attrs(cls, attrs: Mapping[str, object]) -> Mapping[str, object]:
        return {key: value for key, value in attrs.items() if key not in {"result_type", "effect"}}

    @classmethod
    def verify(cls, node: Node, module: IRModule) -> None:
        cls.verify_arity(node)
        required = {"semantic_op", "candidate", "parameters", "facts", "semantic_attrs"}
        if set(node.attrs) != required:
            raise IRVerificationError(f"tir.kernel requires attributes {sorted(required)}.", node_id=node.id)

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(flops=None, bytes_read=None, bytes_written=None, notes=(str(node.attrs["candidate"]),))

    @classmethod
    def python_call(cls, node: Node) -> PythonCall:
        keywords: dict[str, object] = {
            "result_type": node.type,
            **dict(node.attrs),
            "name": node.id,
        }
        if node.effect != PURE:
            keywords["effect"] = node.effect
        if node.metadata:
            keywords["metadata"] = node.metadata
        return PythonCall("F.tir.kernel", tuple(NodeRef(value) for value in node.inputs), keywords)
