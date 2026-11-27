# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""L2 normalization with explicit reduction axes and denominator semantics.

Fast CUDA arithmetic uses sqrt.approx.ftz and div.full. CPU evaluation is a
mathematical reference for this mode, rather than a CUDA instruction oracle.
"""

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
    reduction_group_size = attribute_parameter(default=1)
    math_mode = attribute_parameter(default="fast")

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
        group = attrs["reduction_group_size"]
        if type(group) is not int or group <= 0 or group & (group - 1):
            raise IRSchemaError("L2Normalization reduction_group_size must be a positive power of two.")
        if group == 1:
            attrs.pop("reduction_group_size")
        if attrs["math_mode"] not in ("fast", "accurate"):
            raise IRSchemaError("L2Normalization math_mode must be fast or accurate.")
        if attrs["math_mode"] == "fast":
            attrs.pop("math_mode")
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
        group = attrs.get("reduction_group_size", 1)
        if group > 1:
            extents = tuple(value.shape[axis].fixed_value for axis in axes)
            if any(extent is None for extent in extents) or math.prod(extents) % group:
                raise IRSchemaError("L2Normalization reduction_group_size must divide the fixed reduction capacity.")
        if isinstance(source, DistributedType) and (
                source.partial is not None or any(source.axis_policies[axis] != SBP.broadcast() for axis in axes)):
            raise IRSchemaError("L2Normalization requires materialized broadcast reduction axes.")
        return source

    @classmethod
    def evaluate(cls, node, arguments, context):
        torch = context.torch
        value = cls.value.read(arguments)
        wide = value.float()
        group = node.attrs.get("reduction_group_size", 1)
        if group == 1:
            square_sum = wide.square().sum(dim=node.attrs["axes"], keepdim=True)
        else:
            axes = tuple(normalize_axis(axis, value.ndim) for axis in node.attrs["axes"])
            outer = tuple(axis for axis in range(value.ndim) if axis not in axes)
            order = (*outer, *axes)
            outer_shape = tuple(value.shape[axis] for axis in outer)
            capacity = math.prod(value.shape[axis] for axis in axes)
            grouped = wide.permute(order).reshape(*outer_shape, capacity // group, group)
            total = torch.zeros_like(grouped[..., 0])
            for column in range(group):
                total = total + grouped[..., column].square()
            count = total.shape[-1]
            tile = 1 << (count - 1).bit_length()
            total = torch.nn.functional.pad(total, (0, tile - count))
            lanes = torch.arange(tile, device=total.device)
            mask = tile // 2
            while mask:
                total = total + total.index_select(-1, lanes ^ mask)
                mask //= 2
            square_sum = total[..., :1].reshape(*outer_shape, *((1,) * len(axes)))
            square_sum = square_sum.permute(tuple(order.index(axis) for axis in range(value.ndim)))
        epsilon = node.attrs["epsilon"]
        stabilized = square_sum + epsilon if node.attrs["epsilon_mode"] == "add" else square_sum.clamp_min(epsilon)
        if value.is_cuda and node.attrs.get("math_mode", "fast") == "fast":
            from triton.flagmega.evaluator.cuda_l2_math import normalize
            return normalize(wide, stabilized,
                             reciprocal=node.attrs["division_mode"] == "reciprocal_multiply").to(value.dtype)
        denominator = torch.sqrt(stabilized)
        result = (wide * (1. / denominator) if node.attrs["division_mode"] == "reciprocal_multiply"
                  else wide / denominator)
        return result.to(value.dtype)

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_written=tensor_nbytes(tensor_of(node.type)),
                      notes=("l2-fp32-square-reduction", "explicit-denominator-and-division"))


__all__ = ["L2Normalization"]
