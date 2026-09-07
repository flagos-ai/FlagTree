# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nncase-compatible rotary position embedding application."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import all_broadcast, placement_of, tensor_of
from triton.flagmega.ir.distributed_type import SBPBroadCast
from triton.flagmega.ir.model import DistributedType, IRType, Node
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    input_parameter,
    op_definition,
    tensor_elements,
    tensor_nbytes,
)
from triton.flagmega.ir.type_pattern import has_rank, is_tensor


@op_definition("nn.rope", namespace="nn", functional_name="rope", display_name="NN.RoPE")
class RoPE(OpDefinition):
    input = input_parameter(is_tensor() & has_rank(3))
    cos = input_parameter(is_tensor() & has_rank(3))
    sin = input_parameter(is_tensor() & has_rank(3))

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        if attrs:
            raise IRSchemaError("RoPE does not accept attributes.")
        value_type = cls.input.type_of(inputs)
        cosine_type = cls.cos.type_of(inputs)
        sine_type = cls.sin.type_of(inputs)
        value = tensor_of(value_type)
        cosine = tensor_of(cosine_type)
        sine = tensor_of(sine_type)
        if cosine.shape != sine.shape:
            raise IRSchemaError("RoPE cos and sin must have identical shapes.")
        if cosine.shape[-1] != value.shape[-1]:
            raise IRSchemaError("RoPE cos/sin must match the input head dimension.")
        for source, target in zip(cosine.shape[:-1], value.shape[:-1]):
            if source != target and source.fixed_value != 1:
                raise IRSchemaError("RoPE cos/sin are not broadcastable to the input.")
        if value.shape[-1].is_fixed and value.shape[-1].fixed_value % 2:
            raise IRSchemaError("RoPE head dimension must be even.")
        placement = placement_of(value_type, cosine_type, sine_type)
        if placement is None:
            return value_type
        if not all(
            isinstance(item, DistributedType)
            for item in (value_type, cosine_type, sine_type)
        ):
            raise IRSchemaError("Distributed RoPE requires every tensor operand to name a placement.")
        assert isinstance(value_type, DistributedType)
        assert isinstance(cosine_type, DistributedType)
        assert isinstance(sine_type, DistributedType)
        if any(item.partial is not None for item in (value_type, cosine_type, sine_type)):
            raise IRSchemaError("RoPE requires materialized distributed operands.")
        if not isinstance(value_type.axis_policies[-1], SBPBroadCast):
            raise IRSchemaError(
                "RoPE rotated dimension cannot be split without an explicit exchange.")
        if not all_broadcast(cosine_type) or not all_broadcast(sine_type):
            raise IRSchemaError("RoPE rotary tables must be broadcast for head-sharded input.")
        return value_type

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.input.read(arguments)
        cosine = cls.cos.read(arguments).to(dtype=value.dtype)
        sine = cls.sin.read(arguments).to(dtype=value.dtype)
        half = value.shape[-1] // 2
        rotated = context.torch.cat((-value[..., half:], value[..., :half]), dim=-1)
        return value * cosine + rotated * sine

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        elements = tensor_elements(node.type)
        size = tensor_nbytes(node.type)
        return OpCost(
            flops=None if elements is None else elements * 3,
            bytes_read=None if size is None else size * 3,
            bytes_written=size,
        )


__all__ = ["RoPE"]
