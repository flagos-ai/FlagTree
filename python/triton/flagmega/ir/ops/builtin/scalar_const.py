# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Rank-zero values used for imported function arguments."""

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.ir.model import IRModule, IRType, Node, TensorType
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    PythonCall,
    attribute_parameter,
    op_definition,
)


@op_definition(
    "builtin.scalar_const",
    namespace="builtin",
    functional_name="scalar_const",
    display_name="ScalarConst",
)
class ScalarConst(OpDefinition):
    result_type = attribute_parameter(positional=True)
    value = attribute_parameter(positional=True)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        result_type = cls.result_type.read((), attrs)
        value = cls.value.read((), attrs)
        if not isinstance(result_type, TensorType) or result_type.rank != 0:
            raise IRSchemaError("F.builtin.scalar_const requires a rank-zero TensorType.")
        if not isinstance(value, (bool, int, float)):
            raise IRSchemaError("F.builtin.scalar_const value must be bool or numeric.")
        return {"result_type": result_type, "value": value}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        return cls.result_type.read(inputs, attrs)

    @classmethod
    def ir_attrs(cls, attrs: Mapping[str, object]) -> Mapping[str, object]:
        return {"value": cls.value.read((), attrs)}

    @classmethod
    def verify(cls, node: Node, module: IRModule) -> None:
        cls.verify_arity(node)
        if (
            set(node.attrs) != {"value"}
            or not isinstance(node.type, TensorType)
            or node.type.rank != 0
        ):
            raise IRVerificationError(
                "builtin.scalar_const requires one value and a rank-zero tensor type.",
                node_id=node.id,
            )

    @classmethod
    def evaluate(cls, node, arguments, context):
        return context.torch.tensor(
            node.attrs["value"], dtype=context.torch_dtype(node.type.dtype)
        )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(notes=("compile-time-scalar",))

    @classmethod
    def python_call(cls, node: Node) -> PythonCall:
        keywords: dict[str, object] = {"name": node.id}
        if node.metadata:
            keywords["metadata"] = node.metadata
        return PythonCall(
            "F.builtin.scalar_const",
            (node.type, node.attrs["value"]),
            keywords,
        )


__all__ = ["ScalarConst"]
