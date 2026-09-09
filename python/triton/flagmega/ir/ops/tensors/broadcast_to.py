# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit broadcasting keeps elementwise operand ABIs unambiguous."""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import DistributedType, SBP, tensor_type
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import VectorType


@op_definition("tensors.broadcast_to", namespace="tensors", functional_name="broadcast_to",
               display_name="Tensors.BroadcastTo")
class BroadcastTo(OpDefinition):
    const_evaluable = True
    numpy_materializable = True
    value = input_parameter(is_tensor())
    shape = attribute_parameter()
    output_lanes = attribute_parameter(default=None)

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        shape = attrs["shape"]
        if not isinstance(shape,
                          (tuple, list)) or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in shape):
            raise IRSchemaError("BroadcastTo shape must contain nonnegative integers.")
        lanes = attrs["output_lanes"]
        if lanes is not None:
            if not isinstance(lanes, (tuple, list)) or any(
                    isinstance(lane, bool) or not isinstance(lane, int) or lane <= 0 for lane in lanes):
                raise IRSchemaError("BroadcastTo output_lanes must contain positive integers.")
            lanes = tuple(lanes)
        return {"shape": tuple(shape), "output_lanes": lanes}

    @classmethod
    def ir_attrs(cls, attrs):
        return {key: value for key, value in attrs.items() if key != "output_lanes" or value is not None}

    @classmethod
    def infer_type(cls, inputs, attrs):
        source = cls.value.type_of(inputs)
        value = tensor_of(source)
        input_lanes = getattr(value.dtype, "lanes", ())
        lanes = input_lanes if attrs.get("output_lanes") is None else tuple(attrs["output_lanes"])
        if len(lanes) < len(input_lanes) or any(old not in (1, new)
                                                for old, new in zip(reversed(input_lanes), reversed(lanes))):
            raise IRSchemaError("BroadcastTo vector element lanes are not broadcastable to output_lanes.")
        element_type = value.dtype.elem_type if isinstance(value.dtype, VectorType) else value.dtype
        output = tensor_type(VectorType(element_type, lanes) if lanes else element_type, attrs["shape"])
        offset = output.rank - value.rank
        if offset < 0 or any(old != new and old != tensor_type(value.dtype, (1, )).shape[0]
                             for old, new in zip(value.shape, output.shape[offset:])):
            raise IRSchemaError("BroadcastTo source is not broadcastable to the requested shape.")
        if not isinstance(source, DistributedType):
            return output
        if any(old != new and policy != SBP.broadcast()
               for old, new, policy in zip(value.shape, output.shape[offset:], source.axis_policies)):
            raise IRSchemaError("Expanding a split BroadcastTo axis requires explicit Boxing.")
        return DistributedType(output, (SBP.broadcast(), ) * offset + source.axis_policies, source.placement,
                               source.partial)

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        tensor = tensor_of(node.type)
        lanes = getattr(tensor.dtype, "lanes", ())
        source = tensor_of(context.types[cls.value.read(node.inputs)])
        return value.reshape(_input_physical_shape(value.shape, source, tensor)).expand(
            (*node.attrs["shape"], *lanes)).contiguous()

    @classmethod
    def materialize_numpy(cls, node, arguments, context):
        import numpy as np
        tensor = tensor_of(node.type)
        source = tensor_of(context.types[cls.value.read(node.inputs)])
        value = cls.value.read(arguments)
        return context.as_contiguous(
            np.broadcast_to(value.reshape(_input_physical_shape(value.shape, source, tensor)),
                            (*node.attrs["shape"], *getattr(tensor.dtype, "lanes", ()))))

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_written=tensor_nbytes(node.type), notes=("broadcast", ))


def _input_physical_shape(shape, source, result):
    """Broadcast outer tensor axes and element lanes independently."""
    old_lanes, new_lanes = getattr(source.dtype, "lanes", ()), getattr(result.dtype, "lanes", ())
    return ((1, ) * (result.rank - source.rank) + tuple(shape[:source.rank]) + (1, ) *
            (len(new_lanes) - len(old_lanes)) + old_lanes)
