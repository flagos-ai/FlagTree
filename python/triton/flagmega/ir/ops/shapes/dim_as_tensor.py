# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A dimension expression used as a rank-zero value.

This mirrors nncase's ``IR.Shapes.AsTensor``: a ``Dimension`` is an expression
in its own right, and this op is the boundary where one becomes an ordinary
scalar operand. It lets a value that is a function of symbolic extents -- a
loop's induction value, for instance -- stay an expression instead of being
materialized as stored data and loaded back.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.ir.dim_expr import Dimension, evaluate_dim
from triton.flagmega.ir.logical_expr import LogicalExpr
from triton.flagmega.ir.model import DType, IRModule, IRType, Node, TensorType
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    PythonCall,
    attribute_parameter,
    op_definition,
)


@op_definition(
    "shapes.dim_as_tensor",
    namespace="shapes",
    functional_name="dim_as_tensor",
    display_name="AsTensor",
)
class DimAsTensor(OpDefinition):
    """Expose a dimension expression as a rank-zero tensor value."""

    result_type = attribute_parameter(positional=True)
    expression = attribute_parameter(positional=True)
    deterministic = True
    const_evaluable = False

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        result_type = cls.result_type.read((), attrs)
        expression = cls.expression.read((), attrs)
        if not isinstance(result_type, TensorType) or result_type.rank != 0:
            raise IRSchemaError("F.shapes.dim_as_tensor requires a rank-zero TensorType.")
        if not isinstance(expression, (Dimension, LogicalExpr)):
            raise IRSchemaError(
                "F.shapes.dim_as_tensor requires a Dimension or LogicalExpr.")
        if isinstance(expression, LogicalExpr) and result_type.dtype is not DType.BOOL:
            raise IRSchemaError("A LogicalExpr value must be typed bool.")
        return {"result_type": result_type, "expression": expression}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        return cls.result_type.read(inputs, attrs)

    @classmethod
    def ir_attrs(cls, attrs: Mapping[str, object]) -> Mapping[str, object]:
        return {"expression": cls.expression.read((), attrs)}

    @classmethod
    def verify(cls, node: Node, module: IRModule) -> None:
        cls.verify_arity(node)
        if set(node.attrs) != {"expression"} or not isinstance(node.type, TensorType) \
                or node.type.rank != 0:
            raise IRVerificationError(
                "shapes.dim_as_tensor requires one expression and a rank-zero tensor type.",
                node_id=node.id,
            )
        if not isinstance(node.attrs["expression"], (Dimension, LogicalExpr)):
            raise IRVerificationError(
                "shapes.dim_as_tensor expression must be a Dimension or LogicalExpr.",
                node_id=node.id)

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost.exact_zero(notes=("dimension-expression",))

    @classmethod
    def evaluate(cls, node, arguments, context):
        # Evaluation binds symbols from the enclosing scope, so a reference
        # interpreter resolves the same expression the kernel computes.
        bindings = getattr(context, "dimension_bindings", None) or {}
        expression = node.attrs["expression"]
        value = (expression.evaluate(bindings) if isinstance(expression, LogicalExpr)
                 else evaluate_dim(expression, bindings))
        return context.torch.tensor(value, dtype=context.torch_dtype(node.type.dtype))

    @classmethod
    def python_call(cls, node: Node) -> PythonCall:
        keywords: dict[str, object] = {"name": node.id}
        if node.metadata:
            keywords["metadata"] = node.metadata
        return PythonCall(
            "F.shapes.dim_as_tensor",
            (node.type, node.attrs["expression"]),
            keywords,
        )


__all__ = ["DimAsTensor"]
