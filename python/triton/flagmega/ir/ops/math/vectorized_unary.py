# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nncase-style vectorized unary operation."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import IRType, Node, TensorType
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_elements, tensor_nbytes
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import VectorType


@op_definition(
    "math.vectorized_unary",
    namespace="math",
    functional_name="vectorized_unary",
    display_name="Math.VectorizedUnary",
)
class VectorizedUnary(OpDefinition):
    const_evaluable = True
    value = input_parameter(is_tensor())
    unary_op = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        unary_op = str(attrs["unary_op"])
        if unary_op not in {"silu"}:
            raise IRSchemaError(f"Unsupported vectorized unary op {unary_op!r}.")
        return {"unary_op": unary_op}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value_type = cls.value.type_of(inputs)
        if not isinstance(tensor_of(value_type).dtype, VectorType):
            raise IRSchemaError("F.math.vectorized_unary requires a VectorType tensor.")
        return value_type

    @classmethod
    def evaluate(cls, node, arguments, context):
        return context.torch.nn.functional.silu(cls.value.read(arguments))

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        assert isinstance(node.type, TensorType)
        elements = tensor_elements(node.type)
        size = tensor_nbytes(node.type)
        return OpCost(flops=None if elements is None else elements * 4, bytes_read=size, bytes_written=size)


__all__ = ["VectorizedUnary"]
