# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Runtime-index gathering along one axis (torch.gather semantics)."""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.axis import normalize_axis
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import SBP, DistributedType, tensor_type
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import DType


@op_definition("tensors.gather", namespace="tensors", functional_name="gather", display_name="Tensors.Gather")
class Gather(OpDefinition):
    """Select elements of ``value`` along ``axis`` with per-row ``indices``.

    ``indices`` has the output shape: every non-``axis`` extent matches
    ``value`` and each element addresses one position on ``axis``. This is the
    primitive noaux_tc routing needs to recover unbiased sigmoid scores at
    bias-selected expert ids; the distributed contract below requires
    broadcast operands, which is exactly that replicated-routing shape.
    """

    const_evaluable = True
    value = input_parameter(is_tensor())
    indices = input_parameter(is_tensor())
    axis = attribute_parameter(default=-1)

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        if isinstance(attrs["axis"], bool) or not isinstance(attrs["axis"], int):
            raise IRSchemaError("Gather axis must be an integer.")
        return attrs

    @classmethod
    def infer_type(cls, inputs, attrs):
        source = cls.value.type_of(inputs)
        index_source = cls.indices.type_of(inputs)
        value, indices = tensor_of(source), tensor_of(index_source)
        if value.dtype not in {DType.BFLOAT16, DType.FLOAT32} or indices.dtype not in {DType.INT32, DType.INT64}:
            raise IRSchemaError("Gather requires floating values and int32/int64 indices.")
        axis = normalize_axis(attrs["axis"], value.rank)
        if indices.rank != value.rank:
            raise IRSchemaError(f"Gather indices must have rank {value.rank}, got {indices.rank}.")
        for dim in range(value.rank):
            if dim == axis:
                continue
            if indices.shape[dim] != value.shape[dim]:
                raise IRSchemaError(
                    f"Gather indices extent {indices.shape[dim]} on axis {dim} must match value extent "
                    f"{value.shape[dim]}.")
        if isinstance(source, DistributedType) != isinstance(index_source, DistributedType):
            raise IRSchemaError("Gather requires value and indices to use the same distribution contract.")
        if isinstance(source, DistributedType):
            if not isinstance(index_source, DistributedType) or index_source.placement != source.placement:
                raise IRSchemaError("Gather requires indices distributed like value.")
            if (source.partial is not None or index_source.partial is not None
                    or any(policy != SBP.broadcast() for policy in source.axis_policies)
                    or any(policy != SBP.broadcast() for policy in index_source.axis_policies)):
                raise IRSchemaError("Gather requires broadcast value and indices; insert explicit Boxing.")
            return DistributedType(tensor_type(value.dtype, indices.shape), source.axis_policies, source.placement)
        return tensor_type(value.dtype, indices.shape)

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        indices = cls.indices.read(arguments)
        axis = int(node.attrs["axis"])
        if indices.dtype in (context.torch_dtype(DType.INT32), ):
            indices = indices.to(context.torch_dtype(DType.INT64))
        return value.gather(axis, indices)

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_written=tensor_nbytes(node.type), notes=("runtime-index-gather", ))
