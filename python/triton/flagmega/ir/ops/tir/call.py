# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Typed TIR function call definition.

The callee is a real :class:`Function` in the surrounding module.  Keeping a
call as an IR node, instead of hiding it in codegen metadata, lets editable
``.py``/``.script`` checkpoints preserve and modify device-function
boundaries.
"""

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.ir.model import Effect, IRModule, IRType, Node, PURE, TupleType
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


@op_definition("tir.call", namespace="tir", functional_name="call", display_name="T.Call")
class Call(OpDefinition):
    supports_broadcast_lifting = False
    arguments = variadic_input_parameter(is_ir_type())
    result_type = attribute_parameter()
    callee = attribute_parameter()
    call_effect = attribute_parameter("effect", default=PURE)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        required = {"result_type", "callee"}
        allowed = required | {"effect"}
        if not required.issubset(attributes) or not set(attributes).issubset(allowed):
            raise IRSchemaError(
                "F.tir.call requires result_type and callee, with optional effect."
            )
        result_type = attributes["result_type"]
        effect = attributes.get("effect", PURE)
        callee = str(attributes["callee"])
        if not isinstance(result_type, IRType):
            raise IRSchemaError("F.tir.call result_type must be an IRType.")
        if not callee:
            raise IRSchemaError("F.tir.call callee must be non-empty.")
        if not isinstance(effect, Effect):
            raise IRSchemaError("F.tir.call effect must be an Effect.")
        return {"result_type": result_type, "callee": callee, "effect": effect}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        return cls.result_type.read(inputs, attrs)

    @classmethod
    def infer_effect(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> Effect:
        return cls.call_effect.read(inputs, attrs)

    @classmethod
    def ir_attrs(cls, attrs: Mapping[str, object]) -> Mapping[str, object]:
        return {"callee": cls.callee.read((), attrs)}

    @classmethod
    def verify(cls, node: Node, module: IRModule) -> None:
        cls.verify_arity(node)
        if set(node.attrs) != {"callee"} or not str(node.attrs["callee"]):
            raise IRVerificationError(
                "tir.call requires exactly one non-empty callee attribute.",
                node_id=node.id,
            )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(notes=(f"call @{node.attrs['callee']}",))

    @classmethod
    def python_call(cls, node: Node) -> PythonCall:
        keywords: dict[str, object] = {
            "callee": node.attrs["callee"],
            "result_type": node.type,
            "name": node.id,
        }
        if node.effect != PURE:
            keywords["effect"] = node.effect
        if node.metadata:
            keywords["metadata"] = node.metadata
        return PythonCall(
            "F.tir.call",
            tuple(NodeRef(value) for value in node.inputs),
            keywords,
        )


def function_result_type(module: IRModule, function_name: str) -> IRType:
    """Return the value type seen by a call to ``function_name``."""

    function = module.function_map[function_name]
    outputs = tuple(module.node_map[value].type for value in function.outputs)
    if len(outputs) == 1:
        return outputs[0]
    return TupleType(outputs)


__all__ = ["Call", "function_result_type"]
