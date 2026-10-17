# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Per-head Q absorption and V expansion with one BF16 rounding boundary."""

from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import SBPBroadCast
from triton.flagmega.ir.model import DistributedType, DType, NoneType, VectorType, tensor_type
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.ops.nn.dsa_indexer import _dsa_broadcast_choices
from triton.flagmega.ir.type_pattern import is_none, is_tensor


@op_definition("nn.mla_projection", namespace="nn", functional_name="mla_projection", display_name="NN.MLAProjection")
class MlaProjection(OpDefinition):
    const_evaluable = True
    value = input_parameter(is_tensor())
    weight = input_parameter(is_tensor())
    scale = input_parameter(is_tensor() | is_none())
    direction = attribute_parameter()
    qk_head_dim = attribute_parameter()
    value_head_dim = attribute_parameter()
    latent_dim = attribute_parameter()
    weight_block_n = attribute_parameter(default=128)
    weight_block_k = attribute_parameter(default=128)
    output_lanes = attribute_parameter(default=())

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        if attrs["direction"] not in {"absorb_query", "expand_value"}:
            raise IRSchemaError("MLA projection direction must be absorb_query or expand_value.")
        for key in ("qk_head_dim", "value_head_dim", "latent_dim", "weight_block_n", "weight_block_k"):
            if isinstance(attrs[key], bool) or not isinstance(attrs[key], int) or attrs[key] <= 0:
                raise IRSchemaError(f"MLA projection {key} must be a positive integer.")
        lanes = attrs["output_lanes"]
        if not isinstance(lanes, (tuple, list)) or any(
            isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in lanes
        ):
            raise IRSchemaError("MLA projection output_lanes must be positive integer components.")
        attrs["output_lanes"] = tuple(lanes)
        return attrs

    @classmethod
    def infer_type(cls, inputs, attrs):
        source_type = cls.value.type_of(inputs)
        for source in inputs:
            kind = source.type
            if isinstance(kind, NoneType):
                continue
            if isinstance(kind, DistributedType) != isinstance(source_type, DistributedType):
                raise IRSchemaError("MLA projection tensor operands must share one distribution.")
            if isinstance(kind, DistributedType) and (
                kind.placement != source_type.placement
                or kind.partial is not None
                or kind.exclusive is not None
                or any(not isinstance(policy, SBPBroadCast) for policy in kind.axis_policies)
            ):
                raise IRSchemaError("MLA projection currently requires fully broadcast tensor operands.")
        x, w = (tensor_of(cls.value.type_of(inputs)), tensor_of(cls.weight.type_of(inputs)))
        scalar = x.dtype.elem_type if isinstance(x.dtype, VectorType) else x.dtype
        if (
            x.rank != 3
            or scalar != DType.BFLOAT16
            or w.rank != 2
            or w.dtype not in {DType.BFLOAT16, DType.FLOAT8_E4M3FN}
        ):
            raise IRSchemaError("MLA projection requires BF16 rank-3 values and scalar BF16/FP8 rank-2 weights.")
        if any(not d.is_fixed for d in (*x.shape, *w.shape)):
            raise IRSchemaError("MLA projection requires static dimensions.")
        tokens, heads, width = (d.fixed_value for d in x.shape)
        width *= x.dtype.lane_count if isinstance(x.dtype, VectorType) else 1
        latent = w.shape[1].fixed_value
        if latent != attrs["latent_dim"]:
            raise IRSchemaError("MLA projection latent_dim disagrees with weight columns.")
        if w.shape[0].fixed_value != heads * (attrs["qk_head_dim"] + attrs["value_head_dim"]):
            raise IRSchemaError("MLA projection interleaved weight rows disagree with head dimensions.")
        expected = attrs["qk_head_dim"] if attrs["direction"] == "absorb_query" else latent
        if width != expected:
            raise IRSchemaError("MLA projection input dimension disagrees with its direction.")
        scale = cls.scale.type_of(inputs)
        if w.dtype == DType.BFLOAT16:
            if not isinstance(scale, NoneType):
                raise IRSchemaError("BF16 MLA projection requires scale=None.")
        else:
            if isinstance(scale, NoneType):
                raise IRSchemaError("FP8 MLA projection requires FP32 block scales.")
            scale = tensor_of(scale)
            rows = (w.shape[0].fixed_value + attrs["weight_block_n"] - 1) // attrs["weight_block_n"]
            cols = (latent + attrs["weight_block_k"] - 1) // attrs["weight_block_k"]
            if (
                scale.dtype != DType.FLOAT32
                or scale.rank != 2
                or any(not d.is_fixed for d in scale.shape)
                or tuple(d.fixed_value for d in scale.shape) != (rows, cols)
            ):
                raise IRSchemaError("MLA projection scale shape must match the FP8 block grid.")
        output_dim = latent if attrs["direction"] == "absorb_query" else attrs["value_head_dim"]
        lanes = attrs["output_lanes"]
        if output_dim % prod(lanes):
            raise IRSchemaError("MLA projection output dimension is not divisible by output_lanes.")
        dtype = VectorType(DType.BFLOAT16, lanes) if lanes else DType.BFLOAT16
        result = tensor_type(dtype, (tokens, heads, output_dim // prod(lanes)))
        return (
            DistributedType(result, source_type.axis_policies, source_type.placement)
            if isinstance(source_type, DistributedType)
            else result
        )

    @classmethod
    def distributed_input_type_tuples(cls, choices, attrs):
        tensor_indices = tuple(i for i, types in enumerate(choices) if not all(isinstance(t, NoneType) for t in types))
        yield from _dsa_broadcast_choices(choices, tensor_indices)

    @classmethod
    def evaluate(cls, node, arguments, context):
        torch = context.torch
        source_type = tensor_of(context.types[cls.value.read(node.inputs)])
        x = cls.value.read(arguments).reshape(source_type.shape[0].fixed_value, source_type.shape[1].fixed_value, -1)
        weight = cls.weight.read(arguments)
        scale = cls.scale.read(arguments)
        if scale is not None:
            expanded = scale.repeat_interleave(node.attrs["weight_block_n"], 0).repeat_interleave(
                node.attrs["weight_block_k"], 1
            )
            weight = (weight.float() * expanded[: weight.shape[0], : weight.shape[1]]).bfloat16()
        heads = x.shape[1]
        qk, value_dim = node.attrs["qk_head_dim"], node.attrs["value_head_dim"]
        weight = weight.reshape(heads, qk + value_dim, -1)
        selected = weight[:, :qk] if node.attrs["direction"] == "absorb_query" else weight[:, qk:].transpose(1, 2)
        result = torch.bmm(x.transpose(0, 1).float(), selected.float()).bfloat16().transpose(0, 1).contiguous()
        lanes = node.attrs["output_lanes"]
        return result.reshape(*result.shape[:-1], result.shape[-1] // prod(lanes), *lanes) if lanes else result

    @classmethod
    def cost(cls, node):
        result = tensor_of(node.type)
        tokens, heads = (d.fixed_value for d in result.shape[:2])
        output = result.shape[-1].fixed_value * (result.dtype.lane_count if isinstance(result.dtype, VectorType) else 1)
        reduction = node.attrs["qk_head_dim"] if node.attrs["direction"] == "absorb_query" else node.attrs["latent_dim"]
        return OpCost(
            flops=2 * tokens * heads * output * reduction,
            bytes_written=tensor_nbytes(node.type),
            notes=("mla-per-head-projection",),
        )
