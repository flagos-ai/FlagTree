# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit sum reduction with optional retained dimensions and packed output."""

from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import DistributedType, SBP, tensor_type
from triton.flagmega.ir.distributed_type import SBPSplit, scale_split_units
from triton.flagmega.ir.dim_expr import try_div_exactly
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.axis import normalize_axis
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import DType, VectorType


@op_definition("math.reduce_sum", namespace="math", functional_name="reduce_sum", display_name="Math.ReduceSum")
class ReduceSum(OpDefinition):
    """Sum tensor axes in FP32, retaining vector components independently.

    Vector lanes are components of each tensor element, not additional tensor
    axes. A negative axis refers to the logical tensor rank. Packed inputs
    retain their complete lane shape; output_lanes may prepend complete final
    tensor-axis groups to that shape. It cannot split or reorder input lanes.
    Scalar inputs retain the existing optional final-axis packing behavior.
    """
    const_evaluable = True
    value = input_parameter(is_tensor())
    axes = attribute_parameter(default=(-1, ))
    keep_dims = attribute_parameter(default=True)
    output_lanes = attribute_parameter(default=())

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        if not isinstance(attrs["axes"],
                          (tuple, list)) or any(isinstance(x, bool) or not isinstance(x, int) for x in attrs["axes"]):
            raise IRSchemaError("ReduceSum axes must be an integer sequence.")
        if not isinstance(attrs["keep_dims"], bool):
            raise IRSchemaError("ReduceSum keep_dims must be boolean.")
        lanes = attrs["output_lanes"]
        if not isinstance(lanes, (tuple, list)) or any(
                isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in lanes):
            raise IRSchemaError("ReduceSum output_lanes must be a sequence of positive integers.")
        return {"axes": tuple(attrs["axes"]), "keep_dims": attrs["keep_dims"], "output_lanes": tuple(lanes)}

    @classmethod
    def ir_attrs(cls, attrs):
        return {key: value for key, value in attrs.items() if key != "output_lanes" or value}

    @classmethod
    def infer_type(cls, inputs, attrs):
        source = cls.value.type_of(inputs)
        value = tensor_of(source)
        packed = isinstance(value.dtype, VectorType)
        scalar_dtype = value.dtype.elem_type if packed else value.dtype
        if scalar_dtype not in {DType.BFLOAT16, DType.FLOAT32}:
            raise IRSchemaError("ReduceSum requires BF16/FP32 scalar or vector elements.")
        axes = tuple(normalize_axis(axis, value.rank) for axis in attrs["axes"])
        if len(set(axes)) != len(axes):
            raise IRSchemaError("ReduceSum axes must be distinct.")
        shape = tuple(1 if axis in axes else dim
                      for axis, dim in enumerate(value.shape)
                      if attrs["keep_dims"] or axis not in axes)
        lanes = tuple(attrs.get("output_lanes", ()))
        input_lanes = value.dtype.lanes if packed else ()
        if packed and lanes and (len(lanes) < len(input_lanes) or lanes[-len(input_lanes):] != input_lanes):
            raise IRSchemaError("ReduceSum packed output must retain its complete input lane shape.")
        groups = prod(lanes[:len(lanes) - len(input_lanes)]) if lanes else 1
        if lanes and (not packed or groups != 1):
            final_dim = try_div_exactly(shape[-1], groups) if shape else None
            if final_dim is None:
                raise IRSchemaError("ReduceSum final output dimension must be divisible by output_lanes.")
            shape = (*shape[:-1], final_dim)
        output = tensor_type(VectorType(scalar_dtype, lanes) if lanes else value.dtype, shape)
        if not isinstance(source, DistributedType):
            return output
        if source.partial is not None:
            raise IRSchemaError("ReduceSum requires materialized input.")
        partial_axes = tuple(
            sorted(mesh_axis for axis in axes if isinstance(source.axis_policies[axis], SBPSplit)
                   for mesh_axis in source.axis_policies[axis].hierarchy_axes))
        policies = tuple(SBP.broadcast() if axis in axes else policy
                         for axis, policy in enumerate(source.axis_policies)
                         if attrs["keep_dims"] or axis not in axes)
        if lanes and groups != 1 and isinstance(policies[-1], SBPSplit):
            final_policy = scale_split_units(policies[-1], 1, groups)
            if final_policy is None:
                raise IRSchemaError("ReduceSum output split must align with output_lanes.")
            policies = (*policies[:-1], final_policy)
        return DistributedType(output, policies, source.placement, SBP.partial(partial_axes) if partial_axes else None)

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        source = tensor_of(context.types[cls.value.read(node.inputs)])
        packed = isinstance(source.dtype, VectorType)
        axes = tuple(normalize_axis(axis, source.rank) for axis in node.attrs["axes"])
        output = (value.float().sum(dim=axes, keepdim=node.attrs["keep_dims"]).to(value.dtype)
                  if node.attrs["axes"] else value)
        lanes = tuple(node.attrs.get("output_lanes", ()))
        if not lanes:
            return output
        if not packed:
            return output.reshape(*output.shape[:-1], output.shape[-1] // prod(lanes), *lanes)
        groups = prod(lanes[:len(lanes) - len(source.dtype.lanes)])
        if lanes == source.dtype.lanes:
            return output
        rank = output.ndim - len(source.dtype.lanes)
        if rank == 0:
            return output.reshape(*lanes)
        return output.reshape(*output.shape[:rank - 1], output.shape[rank - 1] // groups, *lanes)

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_written=tensor_nbytes(node.type), notes=("sum-fp32-accumulation", ))
