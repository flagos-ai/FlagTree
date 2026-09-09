# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nncase-compatible rotary position embedding application."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.dim_expr import DimConst
from triton.flagmega.ir.distributed_inference import all_broadcast, placement_of, tensor_of
from triton.flagmega.ir.distributed_type import SBPBroadCast
from triton.flagmega.ir.model import DistributedType, IRType, Node
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
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
    # Scalar coordinates, independent of a later typed-vector representation.
    # None keeps the historical full-head RoPE contract.
    rotary_dim = attribute_parameter(default=None)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        cls.validate_rotary_dim(attrs.get("rotary_dim"))
        return {key: value for key, value in attrs.items() if value is not None}

    @staticmethod
    def validate_rotary_dim(rotary_dim):
        if rotary_dim is not None and (
            isinstance(rotary_dim, bool) or not isinstance(rotary_dim, int)
            or rotary_dim <= 0 or rotary_dim % 2
        ):
            raise IRSchemaError("RoPE rotary_dim must be a positive even integer or None.")

    @classmethod
    def rotary_extent(cls, head_dim, attrs):
        rotary_dim = attrs.get("rotary_dim")
        cls.validate_rotary_dim(rotary_dim)
        if rotary_dim is None:
            if head_dim.is_fixed and head_dim.fixed_value % 2:
                raise IRSchemaError("RoPE head dimension must be even.")
            return head_dim
        if head_dim.minimum is None or head_dim.minimum < rotary_dim:
            raise IRSchemaError("RoPE rotary_dim must not exceed the minimum input head dimension.")
        return DimConst(rotary_dim)

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        attrs = cls.normalize_attrs(attrs)
        value_type = cls.input.type_of(inputs)
        cosine_type = cls.cos.type_of(inputs)
        sine_type = cls.sin.type_of(inputs)
        value = tensor_of(value_type)
        cosine = tensor_of(cosine_type)
        sine = tensor_of(sine_type)
        if cosine.shape != sine.shape:
            raise IRSchemaError("RoPE cos and sin must have identical shapes.")
        if cosine.shape[-1] != cls.rotary_extent(value.shape[-1], attrs):
            raise IRSchemaError("RoPE cos/sin must match the rotary dimension.")
        for source, target in zip(cosine.shape[:-1], value.shape[:-1]):
            if source != target and source.value != 1:
                raise IRSchemaError("RoPE cos/sin are not broadcastable to the input.")
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
        return cls.apply_rotary(value, cls.cos.read(arguments), cls.sin.read(arguments),
                                node.attrs.get("rotary_dim"), context.torch)

    @staticmethod
    def apply_rotary(value, cosine, sine, rotary_dim, torch):
        extent = value.shape[-1] if rotary_dim is None else rotary_dim
        # Like nncase, storage dtypes do not introduce intermediate rounding.
        # In particular, an FP32 rotary table must not first pass through BF16.
        prefix = value[..., :extent].float()
        half = extent // 2
        rotated = torch.cat((-prefix[..., half:], prefix[..., :half]), dim=-1)
        result = (prefix * cosine.float() + rotated * sine.float()).to(value.dtype)
        if extent == value.shape[-1]:
            return result
        return torch.cat((result, value[..., extent:]), dim=-1)

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        output = tensor_of(node.type)
        elements = tensor_elements(output)
        size = tensor_nbytes(output)
        head = output.shape[-1].value
        rotary = cls.rotary_extent(output.shape[-1], node.attrs).value
        rotated_elements = None if elements is None or head is None or not head else elements // head * rotary
        return OpCost(
            flops=None if rotated_elements is None else rotated_elements * 3,
            bytes_read=None if size is None else size * 3,
            bytes_written=size,
        )


__all__ = ["RoPE"]
