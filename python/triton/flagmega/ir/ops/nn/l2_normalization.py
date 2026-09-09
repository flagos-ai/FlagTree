# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""L2 normalization with explicit reduction axes and denominator semantics."""

import math

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.axis import normalize_axis
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import DistributedType, SBP
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import DType


@op_definition("nn.l2_normalization", namespace="nn", functional_name="l2_normalization",
               display_name="NN.L2Normalization")
class L2Normalization(OpDefinition):
    const_evaluable = True
    value = input_parameter(is_tensor())
    axes = attribute_parameter(default=(-1,))
    epsilon = attribute_parameter(default=1e-10)
    epsilon_mode = attribute_parameter(default="clamp")
    division_mode = attribute_parameter(default="divide")

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        axes = attrs["axes"]
        if (not isinstance(axes, (tuple, list)) or not axes
                or any(type(axis) is not int for axis in axes)):
            raise IRSchemaError("L2Normalization axes must be a nonempty sequence of integers.")
        epsilon = attrs["epsilon"]
        if (isinstance(epsilon, bool) or not isinstance(epsilon, (int, float))
                or not math.isfinite(epsilon) or epsilon <= 0):
            raise IRSchemaError("L2Normalization epsilon must be finite and positive.")
        if attrs["epsilon_mode"] not in ("add", "clamp"):
            raise IRSchemaError("L2Normalization epsilon_mode must be add or clamp.")
        if attrs["division_mode"] not in ("divide", "reciprocal_multiply"):
            raise IRSchemaError("L2Normalization division_mode must be divide or reciprocal_multiply.")
        return {**attrs, "axes": tuple(axes), "epsilon": float(epsilon)}

    @classmethod
    def infer_type(cls, inputs, attrs):
        source = cls.value.type_of(inputs)
        value = tensor_of(source)
        if value.dtype not in (DType.BFLOAT16, DType.FLOAT32):
            raise IRSchemaError("L2Normalization requires scalar BF16/FP32 elements.")
        axes = tuple(normalize_axis(axis, value.rank) for axis in attrs["axes"])
        if len(set(axes)) != len(axes):
            raise IRSchemaError("L2Normalization reduction axes must be unique.")
        if isinstance(source, DistributedType) and (
                source.partial is not None or any(source.axis_policies[axis] != SBP.broadcast() for axis in axes)):
            raise IRSchemaError("L2Normalization requires materialized broadcast reduction axes.")
        return source

    @classmethod
    def evaluate(cls, node, arguments, context):
        torch = context.torch
        value = cls.value.read(arguments)
        wide = value.float()
        square_sum = wide.square().sum(dim=node.attrs["axes"], keepdim=True)
        epsilon = node.attrs["epsilon"]
        denominator = torch.sqrt(square_sum + epsilon if node.attrs["epsilon_mode"] == "add"
                                 else square_sum.clamp_min(epsilon))
        result = (wide * (1. / denominator) if node.attrs["division_mode"] == "reciprocal_multiply"
                  else wide / denominator)
        return result.to(value.dtype)

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_written=tensor_nbytes(tensor_of(node.type)),
                      notes=("l2-fp32-square-reduction", "explicit-denominator-and-division"))


__all__ = ["L2Normalization"]
