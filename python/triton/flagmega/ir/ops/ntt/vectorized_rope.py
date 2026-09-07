# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""RoPE over nncase-compatible typed-vector physical layouts."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import placement_of, tensor_of
from triton.flagmega.ir.distributed_type import SBPBroadCast
from triton.flagmega.ir.model import DistributedType, IRType, Node, TensorType
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    input_parameter,
    op_definition,
    tensor_elements,
    tensor_nbytes,
)
from triton.flagmega.ir.ops.tensors.pack import pack_physical
from triton.flagmega.ir.ops.tensors.unpack import unpack_physical
from triton.flagmega.ir.type_pattern import has_rank, is_tensor
from triton.flagmega.ir.types import DType, VectorType


@op_definition(
    "ntt.vectorized_rope",
    namespace="ntt",
    functional_name="vectorized_rope",
    display_name="NTT.VectorizedRoPE",
)
class VectorizedRoPE(OpDefinition):
    """Apply RoPE with the rotary pair and SIMD lane encoded in VectorType.

    The value carries one lane group on its final logical axis.  Cosine and
    sine carry ``(2, lane)`` on that same axis: the first lane selects the two
    rotary halves and the second is the target vector lane.  This is the
    physical contract produced by nncase ``VectorizeRoPEPropagation``.
    """

    input = input_parameter(is_tensor() & has_rank(3))
    cos = input_parameter(is_tensor() & has_rank(3))
    sin = input_parameter(is_tensor() & has_rank(3))

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        if attrs:
            raise IRSchemaError("VectorizedRoPE does not accept attributes.")
        input_ir = cls.input.type_of(inputs)
        cos_ir = cls.cos.type_of(inputs)
        sin_ir = cls.sin.type_of(inputs)
        input_type = tensor_of(input_ir)
        cos_type = tensor_of(cos_ir)
        sin_type = tensor_of(sin_ir)
        input_vector = input_type.dtype
        if not isinstance(input_vector, VectorType) or len(input_vector.lanes) != 1:
            raise IRSchemaError(
                "VectorizedRoPE input requires one final-axis VectorType lane group."
            )
        expected_table_lanes = (2, input_vector.lanes[0])
        if (
            not isinstance(cos_type.dtype, VectorType)
            or not isinstance(sin_type.dtype, VectorType)
            or cos_type.dtype.lanes != expected_table_lanes
            or sin_type.dtype.lanes != expected_table_lanes
        ):
            raise IRSchemaError(
                "VectorizedRoPE cos and sin require rotary pair and lane groups "
                f"{expected_table_lanes}."
            )
        if cos_type.dtype.elem_type != DType.FLOAT32 or sin_type.dtype.elem_type != DType.FLOAT32:
            raise IRSchemaError("VectorizedRoPE cos and sin must have float32 elements.")
        if cos_type.shape != sin_type.shape:
            raise IRSchemaError("VectorizedRoPE cos and sin must have identical shapes.")
        logical_input_extent = input_type.shape[-1] * input_vector.lanes[0]
        logical_table_extent = cos_type.shape[-1] * (2 * input_vector.lanes[0])
        if logical_input_extent != logical_table_extent:
            raise IRSchemaError(
                "VectorizedRoPE rotary tables must match the unpacked input head dimension."
            )
        for source, target in zip(cos_type.shape[:-1], input_type.shape[:-1]):
            if source != target and source.fixed_value != 1:
                raise IRSchemaError(
                    "VectorizedRoPE rotary tables are not broadcastable to the input."
                )

        placement = placement_of(input_ir, cos_ir, sin_ir)
        if placement is None:
            return input_ir
        if not all(
            isinstance(value, DistributedType) for value in (input_ir, cos_ir, sin_ir)
        ):
            raise IRSchemaError(
                "Distributed VectorizedRoPE requires every operand to name a placement."
            )
        assert isinstance(input_ir, DistributedType)
        assert isinstance(cos_ir, DistributedType)
        assert isinstance(sin_ir, DistributedType)
        if any(value.partial is not None for value in (input_ir, cos_ir, sin_ir)):
            raise IRSchemaError("VectorizedRoPE requires materialized distributed operands.")
        if (
            input_ir.axis_policies[0] != cos_ir.axis_policies[0]
            or not isinstance(cos_ir.axis_policies[1], SBPBroadCast)
            or input_ir.axis_policies[2] != cos_ir.axis_policies[2]
            or cos_ir.axis_policies != sin_ir.axis_policies
            or not isinstance(input_ir.axis_policies[-1], SBPBroadCast)
        ):
            raise IRSchemaError(
                "VectorizedRoPE distributed operands have incompatible axis policies."
            )
        return input_ir

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.input.read(arguments)
        cosine = cls.cos.read(arguments)
        sine = cls.sin.read(arguments)
        input_type = tensor_of(context.types[cls.input.read(node.inputs)])
        cos_type = tensor_of(context.types[cls.cos.read(node.inputs)])
        sin_type = tensor_of(context.types[cls.sin.read(node.inputs)])
        assert isinstance(input_type.dtype, VectorType)
        assert isinstance(cos_type.dtype, VectorType)
        assert isinstance(sin_type.dtype, VectorType)
        rotary_axis = input_type.rank - 1
        scalar_value = unpack_physical(
            value, input_type.rank, input_type.dtype.lanes, (rotary_axis,)
        )
        scalar_cos = unpack_physical(
            cosine,
            cos_type.rank,
            cos_type.dtype.lanes,
            (rotary_axis, rotary_axis),
        ).to(dtype=scalar_value.dtype)
        scalar_sin = unpack_physical(
            sine,
            sin_type.rank,
            sin_type.dtype.lanes,
            (rotary_axis, rotary_axis),
        ).to(dtype=scalar_value.dtype)
        half = scalar_value.shape[-1] // 2
        rotated = context.torch.cat(
            (-scalar_value[..., half:], scalar_value[..., :half]), dim=-1
        )
        result = scalar_value * scalar_cos + rotated * scalar_sin
        return pack_physical(
            result,
            input_type.rank,
            input_type.dtype.lanes,
            (rotary_axis,),
        )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        output = tensor_of(node.type)
        elements = tensor_elements(output)
        size = tensor_nbytes(output)
        return OpCost(
            flops=None if elements is None else elements * 3,
            bytes_read=None if size is None else size * 3,
            bytes_written=size,
            notes=("vectorized-rope",),
        )


__all__ = ["VectorizedRoPE"]
