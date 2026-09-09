# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit sum reduction with optional retained dimensions."""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import DistributedType, SBP, tensor_type
from triton.flagmega.ir.distributed_type import SBPSplit
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.axis import normalize_axis
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import DType


@op_definition("math.reduce_sum", namespace="math", functional_name="reduce_sum", display_name="Math.ReduceSum")
class ReduceSum(OpDefinition):
    const_evaluable = True
    value = input_parameter(is_tensor())
    axes = attribute_parameter(default=(-1, ))
    keep_dims = attribute_parameter(default=True)

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        if not isinstance(attrs["axes"],
                          (tuple, list)) or any(isinstance(x, bool) or not isinstance(x, int) for x in attrs["axes"]):
            raise IRSchemaError("ReduceSum axes must be an integer sequence.")
        if not isinstance(attrs["keep_dims"], bool):
            raise IRSchemaError("ReduceSum keep_dims must be boolean.")
        return {"axes": tuple(attrs["axes"]), "keep_dims": attrs["keep_dims"]}

    @classmethod
    def infer_type(cls, inputs, attrs):
        source = cls.value.type_of(inputs)
        value = tensor_of(source)
        if value.dtype not in {DType.BFLOAT16, DType.FLOAT32}:
            raise IRSchemaError("ReduceSum requires scalar BF16/FP32 elements.")
        axes = tuple(normalize_axis(axis, value.rank) for axis in attrs["axes"])
        if len(set(axes)) != len(axes):
            raise IRSchemaError("ReduceSum axes must be distinct.")
        shape = tuple(1 if axis in axes else dim
                      for axis, dim in enumerate(value.shape)
                      if attrs["keep_dims"] or axis not in axes)
        output = tensor_type(value.dtype, shape)
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
        return DistributedType(output, policies, source.placement, SBP.partial(partial_axes) if partial_axes else None)

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        if not node.attrs["axes"]:
            return value
        return value.float().sum(dim=tuple(node.attrs["axes"]), keepdim=node.attrs["keep_dims"]).to(value.dtype)

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_written=tensor_nbytes(node.type), notes=("sum-fp32-accumulation", ))
