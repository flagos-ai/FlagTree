# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Storage-preserving tensor element reinterpretation."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.dim_expr import try_div_exactly
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import SBPSplit, scale_split_units
from triton.flagmega.ir.model import DistributedType, IRType, Node, TensorType, tensor_type
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
)
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import (
    VectorType,
    data_type,
    data_type_from_data,
    data_type_to_data,
)


@op_definition(
    "tensors.bitcast",
    namespace="tensors",
    functional_name="bitcast",
    display_name="Tensors.Bitcast",
)
class Bitcast(OpDefinition):
    """Reinterpret bytes as ``dtype`` without numeric conversion."""

    const_evaluable = True
    numpy_materializable = True
    value = input_parameter(is_tensor())
    byte_preserving_input_parameters = (value,)
    dtype = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        dtype = attrs["dtype"]
        if isinstance(dtype, dict):
            dtype = data_type_from_data(dtype)
        return {"dtype": data_type_to_data(data_type(dtype))}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        source_type = cls.value.type_of(inputs)
        source = tensor_of(source_type)
        target_dtype = data_type_from_data(attrs["dtype"])
        shape = list(source.shape)
        if source.dtype.itemsize != target_dtype.itemsize:
            scaled_bytes = source.dtype.itemsize
            if not shape:
                extent = try_div_exactly(scaled_bytes, target_dtype.itemsize)
                if extent is None:
                    raise IRSchemaError(
                        "F.tensors.bitcast cannot reinterpret a scalar into a "
                        "larger element type."
                    )
                shape = [extent]
            else:
                extent = try_div_exactly(
                    shape[-1] * scaled_bytes,
                    target_dtype.itemsize,
                )
                if extent is None:
                    raise IRSchemaError(
                        "F.tensors.bitcast requires the final logical extent "
                        "to preserve an integral number of target elements."
                    )
                shape[-1] = extent
        result = tensor_type(target_dtype, shape)
        if not isinstance(source_type, DistributedType):
            return result
        if source_type.partial is not None:
            raise IRSchemaError(
                "F.tensors.bitcast does not support partial distributed values."
            )
        policies = list(source_type.axis_policies)
        if policies and source.dtype.itemsize != target_dtype.itemsize:
            last = len(policies) - 1
            if isinstance(policies[last], SBPSplit):
                scaled = scale_split_units(
                    policies[last], source.dtype.itemsize, target_dtype.itemsize)
                if scaled is None:
                    raise IRSchemaError(
                        "F.tensors.bitcast cannot preserve the final-axis split "
                        "unit for the requested element type."
                    )
                policies[last] = scaled
        return DistributedType(result, tuple(policies), source_type.placement)

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments).contiguous()
        source = tensor_of(context.types[cls.value.read(node.inputs)])
        target = tensor_of(node.type)
        target_dtype = context.torch_dtype(target.dtype)
        reinterpreted = value.view(target_dtype)

        source_outer = tuple(int(value.shape[index]) for index in range(source.rank))
        if source.rank == 0:
            target_outer = () if target.rank == 0 else (
                source.dtype.itemsize // target.dtype.itemsize,
            )
        else:
            target_outer = list(source_outer)
            target_outer[-1] = (
                target_outer[-1]
                * source.dtype.itemsize
                // target.dtype.itemsize
            )
            target_outer = tuple(target_outer)
        lanes = target.dtype.lanes if isinstance(target.dtype, VectorType) else ()
        return reinterpreted.reshape((*target_outer, *lanes))

    @classmethod
    def materialize_numpy(cls, node, arguments, context):
        value = context.as_contiguous(cls.value.read(arguments))
        source = tensor_of(context.types[cls.value.read(node.inputs)])
        target = tensor_of(node.type)
        reinterpreted = value.view(context.storage_dtype(target.dtype))
        source_outer = tuple(int(value.shape[index]) for index in range(source.rank))
        if source.rank == 0:
            target_outer = () if target.rank == 0 else (
                source.dtype.itemsize // target.dtype.itemsize,
            )
        else:
            target_outer = list(source_outer)
            target_outer[-1] = (
                target_outer[-1] * source.dtype.itemsize // target.dtype.itemsize
            )
            target_outer = tuple(target_outer)
        lanes = target.dtype.lanes if isinstance(target.dtype, VectorType) else ()
        return reinterpreted.reshape((*target_outer, *lanes))

    @classmethod
    def python_attrs(cls, node: Node):
        return {"dtype": data_type_from_data(node.attrs["dtype"])}

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(flops=0, bytes_read=0, bytes_written=0, notes=("bitcast-view",))


__all__ = ["Bitcast"]
