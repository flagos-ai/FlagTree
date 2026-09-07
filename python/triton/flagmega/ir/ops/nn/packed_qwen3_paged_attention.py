# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Qwen3 paged attention with an offline split-K K-major QKV asset."""

from __future__ import annotations

from math import prod
from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.model import DType, Effect, IRType, Node, TensorType, TupleType, effect
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    ParameterKind,
    attribute_parameter,
    input_parameter,
    op_definition,
)
from triton.flagmega.ir.ops.nn.qwen3_paged_attention import _scalar_bool, _scalar_int, qwen3_paged_attention
from triton.flagmega.ir.type_pattern import has_dtype, has_rank, is_ref, is_tensor
from triton.flagmega.ir.ops.tensors._k_major import (
    parse_qkv_split_k_layout,
)


@op_definition(
    "nn.packed_qwen3_paged_attention",
    namespace="nn",
    functional_name="packed_qwen3_paged_attention",
    display_name="NN.PackedQwen3PagedAttention",
)
class PackedQwen3PagedAttention(OpDefinition):
    """The semantic attention op after AutoPacking.

    The packed operand is produced by ordinary constant tensor operations.  A
    target kernel may consume it directly, while the evaluator reverses the
    physical layout and executes the same reference semantics as the logical
    operation.
    """

    value = input_parameter(is_tensor() & has_rank(2))
    state = input_parameter(
        is_ref(), parameter_kind=ParameterKind.ATTRIBUTE,
        memory_effect=MemoryEffect.CHIP_READ_WRITE.partitioned_by_argument(6),
    )
    packed_qkv_weight = input_parameter(is_tensor() & has_rank(4))
    q_norm_weight = input_parameter(is_tensor() & has_rank(1))
    k_norm_weight = input_parameter(is_tensor() & has_rank(1))
    output_weight = input_parameter(is_tensor() & has_rank(2))
    layer_id = input_parameter(
        is_tensor() & has_rank(0) & has_dtype(DType.INT32),
        parameter_kind=ParameterKind.ATTRIBUTE,
    )
    advance_sequence = input_parameter(
        is_tensor() & has_rank(0) & has_dtype(DType.BOOL),
        parameter_kind=ParameterKind.ATTRIBUTE,
    )
    num_attention_heads = attribute_parameter()
    num_key_value_heads = attribute_parameter()
    head_dim = attribute_parameter()
    epsilon = attribute_parameter()
    rope_theta = attribute_parameter()
    context_mesh_size = attribute_parameter()
    head_mesh_size = attribute_parameter()
    block_k = attribute_parameter()
    n_lane = attribute_parameter()
    k_lane = attribute_parameter()
    packed_layout = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        for name in (
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "context_mesh_size", "head_mesh_size", "block_k", "n_lane", "k_lane",
        ):
            value = attrs[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise IRSchemaError(f"PackedQwen3PagedAttention {name} is invalid: {value!r}.")
        if attrs["num_attention_heads"] % attrs["num_key_value_heads"]:
            raise IRSchemaError("Qwen3 attention heads must be divisible by KV heads.")
        layout_lanes = parse_qkv_split_k_layout(str(attrs["packed_layout"]))
        if layout_lanes != (attrs["n_lane"], attrs["k_lane"]):
            raise IRSchemaError(
                "PackedQwen3PagedAttention layout identity disagrees with n_lane/k_lane."
            )
        if attrs["block_k"] % attrs["k_lane"]:
            raise IRSchemaError(
                "PackedQwen3PagedAttention block_k must be divisible by k_lane."
            )
        attrs["epsilon"] = float(attrs["epsilon"])
        attrs["rope_theta"] = float(attrs["rope_theta"])
        if attrs["epsilon"] <= 0 or attrs["rope_theta"] <= 0:
            raise IRSchemaError("Qwen3 RMS epsilon and RoPE theta must be positive.")
        return attrs

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value = cls.value.type_of(inputs)
        packed = cls.packed_qkv_weight.type_of(inputs)
        state = cls.state.type_of(inputs)
        output_weight = cls.output_weight.type_of(inputs)
        assert isinstance(value, TensorType) and isinstance(packed, TensorType)
        assert isinstance(output_weight, TensorType)
        hidden = value.shape[-1].fixed_value
        heads = int(attrs["num_attention_heads"])
        kv_heads = int(attrs["num_key_value_heads"])
        head_dim = int(attrs["head_dim"])
        q_size = heads * head_dim
        kv_size = kv_heads * head_dim
        mesh_y = int(attrs["context_mesh_size"])
        mesh_x = int(attrs["head_mesh_size"])
        block_k = int(attrs["block_k"])
        n_lane = int(attrs["n_lane"])
        k_lane = int(attrs["k_lane"])
        if hidden is None or q_size != hidden:
            raise IRSchemaError("Packed QKV requires a static hidden size equal to query size.")
        if hidden % mesh_y or (q_size + 2 * kv_size) % mesh_x:
            raise IRSchemaError("Packed QKV dimensions are not divisible by the selected 2D mesh.")
        k_per_shard = hidden // mesh_y
        local_n = (q_size + 2 * kv_size) // mesh_x
        if k_per_shard % block_k or local_n % n_lane:
            raise IRSchemaError("Packed QKV local N/K dimensions do not tile the selected K-major layout.")
        expected = (
            mesh_y * mesh_x * (k_per_shard // block_k) * (local_n // n_lane),
            block_k // k_lane,
            packed.shape[-2].fixed_value,
            packed.shape[-1].fixed_value,
        )
        actual = tuple(dimension.value for dimension in packed.shape)
        if (
            actual != expected
            or actual[-2] is None
            or actual[-1] is None
            or int(actual[-2]) * int(actual[-1]) != n_lane * k_lane
        ):
            raise IRSchemaError(f"Packed QKV weight shape must be {expected}, got {actual}.")
        if packed.dtype != value.dtype:
            raise IRSchemaError("Packed QKV weight dtype must match the activation dtype.")
        if tuple(dimension.value for dimension in output_weight.shape) != (hidden, q_size):
            raise IRSchemaError("Qwen3 o_proj weight shape does not match attention dimensions.")
        if tuple(dimension.value for dimension in cls.q_norm_weight.type_of(inputs).shape) != (head_dim,):
            raise IRSchemaError("Qwen3 q_norm weight must match head_dim.")
        if tuple(dimension.value for dimension in cls.k_norm_weight.type_of(inputs).shape) != (head_dim,):
            raise IRSchemaError("Qwen3 k_norm weight must match head_dim.")
        return TupleType((value, state))

    @classmethod
    def infer_effect(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> Effect:
        return effect("read_write", "paged_attention_kv_cache")

    @classmethod
    def evaluate(cls, node, arguments, context):
        attrs = node.attrs
        q_weight, k_weight, v_weight = unpack_split_k_qkv_weight(
            cls.packed_qkv_weight.read(arguments),
            hidden_size=cls.value.read(arguments).shape[-1],
            num_attention_heads=int(attrs["num_attention_heads"]),
            num_key_value_heads=int(attrs["num_key_value_heads"]),
            head_dim=int(attrs["head_dim"]),
            context_mesh_size=int(attrs["context_mesh_size"]),
            head_mesh_size=int(attrs["head_mesh_size"]),
            block_k=int(attrs["block_k"]),
            n_lane=int(attrs["n_lane"]),
            k_lane=int(attrs["k_lane"]),
        )
        return qwen3_paged_attention(
            cls.value.read(arguments),
            cls.state.read(arguments),
            q_weight,
            k_weight,
            v_weight,
            cls.q_norm_weight.read(arguments),
            cls.k_norm_weight.read(arguments),
            cls.output_weight.read(arguments),
            layer_id=_scalar_int(cls.layer_id.read(arguments)),
            num_attention_heads=int(attrs["num_attention_heads"]),
            num_key_value_heads=int(attrs["num_key_value_heads"]),
            head_dim=int(attrs["head_dim"]),
            epsilon=float(attrs["epsilon"]),
            rope_theta=float(attrs["rope_theta"]),
            advance_sequence=_scalar_bool(cls.advance_sequence.read(arguments)),
            torch=context.torch,
        )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(flops=None, bytes_read=None, bytes_written=None, notes=("packed-split-k-qkv-attention",))


def unpack_split_k_qkv_weight(
    packed,
    *,
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    context_mesh_size: int,
    head_mesh_size: int,
    block_k: int,
    n_lane: int = 8,
    k_lane: int = 16,
):
    """Reverse the documented physical layout for reference evaluation."""

    q_size = num_attention_heads * head_dim
    kv_size = num_key_value_heads * head_dim
    mesh_size = context_mesh_size * head_mesh_size
    k_per_shard = hidden_size // context_mesh_size
    local_n = (q_size + 2 * kv_size) // head_mesh_size
    k_tiles = k_per_shard // block_k
    logical = packed.reshape(
        mesh_size, k_tiles, local_n // n_lane, block_k // k_lane, n_lane, k_lane)
    logical = logical.permute(0, 1, 2, 4, 3, 5).reshape(
        mesh_size, k_tiles, local_n, block_k)
    logical = logical.permute(0, 2, 1, 3).reshape(
        context_mesh_size, head_mesh_size, local_n, k_per_shard)
    q_local = logical[:, :, : q_size // head_mesh_size, :]
    kv_local = logical[:, :, q_size // head_mesh_size :, :]
    q_weight = q_local.permute(1, 2, 0, 3).reshape(q_size, hidden_size)
    kv_weight = kv_local.permute(1, 2, 0, 3).reshape(2 * kv_size, hidden_size)
    return q_weight, kv_weight[:kv_size], kv_weight[kv_size:]


__all__ = ["PackedQwen3PagedAttention", "unpack_split_k_qkv_weight"]
