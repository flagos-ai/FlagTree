# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nncase-style vectorized binary operation."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import IRType, Node, TensorType
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_elements, tensor_nbytes
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import VectorType


@op_definition(
    "math.vectorized_binary",
    namespace="math",
    functional_name="vectorized_binary",
    display_name="Math.VectorizedBinary",
)
class VectorizedBinary(OpDefinition):
    const_evaluable = True
    lhs = input_parameter(is_tensor())
    rhs = input_parameter(is_tensor())
    binary_op = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        binary_op = str(attrs["binary_op"])
        if binary_op not in {"add", "mul"}:
            raise IRSchemaError(f"Unsupported vectorized binary op {binary_op!r}.")
        return {"binary_op": binary_op}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        lhs = cls.lhs.type_of(inputs)
        rhs = cls.rhs.type_of(inputs)
        lhs_tensor = tensor_of(lhs)
        if lhs != rhs or not isinstance(lhs_tensor.dtype, VectorType):
            raise IRSchemaError("F.math.vectorized_binary requires identical VectorType tensors.")
        return lhs

    @classmethod
    def evaluate(cls, node, arguments, context):
        lhs = cls.lhs.read(arguments)
        rhs = cls.rhs.read(arguments)
        return lhs + rhs if node.attrs["binary_op"] == "add" else lhs * rhs

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        assert isinstance(node.type, TensorType)
        elements = tensor_elements(node.type)
        size = tensor_nbytes(node.type)
        return OpCost(flops=elements, bytes_read=None if size is None else size * 2, bytes_written=size)


__all__ = ["VectorizedBinary"]
