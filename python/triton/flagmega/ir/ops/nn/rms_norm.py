# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""RMSNorm definition and its local behaviors."""

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import all_broadcast, placement_of, tensor_of
from triton.flagmega.ir.distributed_type import SBPBroadCast
from triton.flagmega.ir.model import DistributedType, IRType, Node
from triton.flagmega.ir.type_pattern import has_rank, is_tensor
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_elements,
    tensor_nbytes,
)


@op_definition(
    "nn.rms_norm",
    namespace="nn",
    functional_name="rms_norm",
    display_name="NN.RMSNorm",
)
class RMSNorm(OpDefinition):
    value = input_parameter(is_tensor())
    weight = input_parameter(is_tensor() & has_rank(1))
    epsilon = attribute_parameter()
    weight_bias = attribute_parameter(default=1.0)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        if not {"epsilon"}.issubset(attributes) or not set(attributes).issubset({"epsilon", "weight_bias"}):
            raise IRSchemaError("RMSNorm requires epsilon and optional weight_bias.")
        epsilon = float(attributes["epsilon"])
        if epsilon <= 0:
            raise IRSchemaError("RMSNorm epsilon must be positive.")
        return {"epsilon": epsilon, "weight_bias": float(attributes.get("weight_bias", 1.0))}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value_type = cls.value.type_of(inputs)
        weight_type = cls.weight.type_of(inputs)
        value = tensor_of(value_type)
        weight = tensor_of(weight_type)
        if value.shape[-1] != weight.shape[0]:
            raise IRSchemaError("RMSNorm weight must match the value's last dimension.")
        placement = placement_of(value_type, weight_type)
        if placement is None:
            return value
        if not isinstance(value_type, DistributedType) or not isinstance(weight_type, DistributedType):
            raise IRSchemaError("Distributed RMSNorm requires both operands to name a placement.")
        if value_type.partial is not None or weight_type.partial is not None:
            raise IRSchemaError("RMSNorm requires materialized distributed operands.")
        if not isinstance(value_type.axis_policies[-1], SBPBroadCast):
            raise IRSchemaError(
                "RMSNorm reduction axis cannot be split; use an explicit "
                "NormStats/NormApply decomposition.")
        if not all_broadcast(weight_type):
            raise IRSchemaError(
                "RMSNorm weight must be broadcast when the reduction axis "
                "is materialized locally.")
        return value_type

    @classmethod
    def evaluate(cls, node, arguments, context):
        return rms_norm(
            cls.value.read(arguments),
            cls.weight.read(arguments),
            epsilon=float(cls.epsilon.read(arguments, node.attrs)),
            weight_bias=float(cls.weight_bias.read(arguments, node.attrs)),
        )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        elements = tensor_elements(node.type) if hasattr(node.type, "shape") else None
        size = tensor_nbytes(node.type) if hasattr(node.type, "shape") else None
        return OpCost(flops=None if elements is None else elements * 5, bytes_read=None, bytes_written=size)


def rms_norm(hidden, weight, *, epsilon: float, weight_bias: float = 1.0):
    variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = hidden.float() * (variance + epsilon).rsqrt()
    normalized = normalized.to(dtype=hidden.dtype)
    return normalized * (weight.to(dtype=hidden.dtype) + weight_bias)


__all__ = ["RMSNorm", "rms_norm"]
