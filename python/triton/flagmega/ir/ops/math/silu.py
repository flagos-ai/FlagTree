# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Math.Silu definition and its local behaviors."""

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import IRType, Node
from triton.flagmega.ir.ops.core import (
    OpCost,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_elements,
    tensor_nbytes,
)
from triton.flagmega.ir.ops.pointwise import SameTypePointwiseOp
from triton.flagmega.ir.type_pattern import is_tensor


@op_definition("math.silu", namespace="math", functional_name="silu", display_name="Math.Silu")
class Silu(SameTypePointwiseOp):
    const_evaluable = True
    value = input_parameter(is_tensor())
    math_mode = attribute_parameter(default="fast")

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        mode = attrs["math_mode"]
        if mode not in {"fast", "accurate"}:
            raise IRSchemaError("Silu math_mode must be 'fast' or 'accurate'.")
        return {"math_mode": mode} if mode != "fast" else {}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        if set(attrs) - {"math_mode"}:
            raise IRSchemaError("F.math.silu only accepts math_mode.")
        return cls.value.type_of(inputs)

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        if value.is_cuda and node.attrs.get("math_mode") == "accurate":
            from triton.flagmega.evaluator.cuda_activation_math import accurate_silu
            return accurate_silu(value)
        return context.torch.nn.functional.silu(value)

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        elements = tensor_elements(node.type) if hasattr(node.type, "shape") else None
        size = tensor_nbytes(node.type) if hasattr(node.type, "shape") else None
        return OpCost(flops=None if elements is None else elements * 4, bytes_read=size, bytes_written=size)
