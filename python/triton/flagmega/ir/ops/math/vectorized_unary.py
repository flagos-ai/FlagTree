# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nncase-style vectorized unary operation."""

from __future__ import annotations

from typing import Mapping, Sequence
from dataclasses import replace

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import DistributedType, IRType, Node
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_elements, tensor_nbytes
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import VectorType
from triton.flagmega.ir.ops.math.silu import Silu
from triton.flagmega.ir.ops.math.sigmoid import Sigmoid


@op_definition(
    "math.vectorized_unary",
    namespace="math",
    functional_name="vectorized_unary",
    display_name="Math.VectorizedUnary",
)
class VectorizedUnary(OpDefinition):
    const_evaluable = True
    scalar_definitions = {"silu": Silu, "sigmoid": Sigmoid}
    value = input_parameter(is_tensor())
    unary_op = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        unary_op = str(attrs["unary_op"])
        if unary_op not in cls.scalar_definitions:
            raise IRSchemaError(f"Unsupported vectorized unary op {unary_op!r}.")
        return {"unary_op": unary_op}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value_type = cls.value.type_of(inputs)
        if not isinstance(tensor_of(value_type).dtype, VectorType):
            raise IRSchemaError("F.math.vectorized_unary requires a VectorType tensor.")
        scalar = replace(tensor_of(value_type), dtype=tensor_of(value_type).dtype.elem_type)
        scalar_type = replace(value_type, tensor=scalar) if isinstance(value_type, DistributedType) else scalar
        cls.scalar_definitions[str(attrs["unary_op"])].infer_type((replace(inputs[0], type=scalar_type),), {})
        return value_type

    @classmethod
    def evaluate(cls, node, arguments, context):
        # Reuse the op-local scalar contract, including Sigmoid's FP32
        # computation followed by its declared BF16/FP32 output rounding.
        return cls.scalar_definitions[str(node.attrs["unary_op"])].evaluate(node, arguments, context)

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        tensor = tensor_of(node.type)
        elements = tensor_elements(tensor)
        if elements is not None:
            elements *= tensor.dtype.lane_count
        size = tensor_nbytes(tensor)
        return OpCost(flops=None if elements is None else elements * 4, bytes_read=size, bytes_written=size)


__all__ = ["VectorizedUnary"]
