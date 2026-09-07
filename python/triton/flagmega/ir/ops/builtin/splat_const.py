# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Compressed compile-time tensor whose elements all have one value."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.ir.model import IRModule, IRType, Node, TensorType
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    PythonCall,
    attribute_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.types import VectorType


@op_definition(
    "builtin.splat_const",
    namespace="builtin",
    functional_name="splat_const",
    display_name="SplatConst",
)
class SplatConst(OpDefinition):
    """A first-class constant tensor without embedding repeated payload bytes."""

    constant_source = True
    numpy_materializable = True
    result_type = attribute_parameter(positional=True)
    value = attribute_parameter(positional=True)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        result_type = cls.result_type.read((), attrs)
        value = cls.value.read((), attrs)
        if not isinstance(result_type, TensorType):
            raise IRSchemaError("F.builtin.splat_const requires a TensorType.")
        if any(not dimension.is_fixed for dimension in result_type.shape):
            raise IRSchemaError("F.builtin.splat_const requires a static tensor shape.")
        if not isinstance(value, (bool, int, float)):
            raise IRSchemaError("F.builtin.splat_const value must be bool or numeric.")
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
        if set(node.attrs) != {"value"} or not isinstance(node.type, TensorType):
            raise IRVerificationError(
                "builtin.splat_const requires one value and a tensor result type.",
                node_id=node.id,
            )

    @classmethod
    def evaluate(cls, node, arguments, context):
        del arguments
        dtype = node.type.dtype.elem_type if isinstance(node.type.dtype, VectorType) else node.type.dtype
        shape = tuple(dimension.fixed_value for dimension in node.type.shape)
        if isinstance(node.type.dtype, VectorType):
            shape = (*shape, *node.type.dtype.lanes)
        return context.torch.full(
            shape,
            node.attrs["value"],
            dtype=context.torch_dtype(dtype),
        )

    @classmethod
    def materialize_numpy(cls, node, arguments, context):
        del arguments
        return context.full(node.type, node.attrs["value"])

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(bytes_written=tensor_nbytes(node.type), notes=("compile-time-splat",))

    @classmethod
    def python_call(cls, node: Node) -> PythonCall:
        keywords: dict[str, object] = {"name": node.id}
        if node.metadata:
            keywords["metadata"] = node.metadata
        return PythonCall(
            "F.builtin.splat_const",
            (node.type, node.attrs["value"]),
            keywords,
        )


__all__ = ["SplatConst"]
