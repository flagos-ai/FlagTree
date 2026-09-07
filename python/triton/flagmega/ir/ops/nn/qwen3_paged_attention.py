# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Qwen3 QK-normalized RoPE paged attention and reference semantics."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import EvaluationError, IRSchemaError
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
from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionState
from triton.flagmega.ir.type_pattern import has_dtype, has_rank, is_ref, is_tensor


@op_definition(
    "nn.qwen3_paged_attention",
    namespace="nn",
    functional_name="qwen3_paged_attention",
    display_name="NN.Qwen3PagedAttention",
)
class Qwen3PagedAttention(OpDefinition):
    value = input_parameter(is_tensor() & has_rank(2))
    state = input_parameter(
        is_ref(), parameter_kind=ParameterKind.ATTRIBUTE,
        memory_effect=MemoryEffect.CHIP_READ_WRITE.partitioned_by_argument(8),
    )
    q_weight = input_parameter(is_tensor() & has_rank(2))
    k_weight = input_parameter(is_tensor() & has_rank(2))
    v_weight = input_parameter(is_tensor() & has_rank(2))
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

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        integer_names = ("num_attention_heads", "num_key_value_heads", "head_dim")
        for name in integer_names:
            value = attrs[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise IRSchemaError(f"Qwen3PagedAttention {name} is invalid: {value!r}.")
        if attrs["num_attention_heads"] % attrs["num_key_value_heads"]:
            raise IRSchemaError("Qwen3 attention heads must be divisible by KV heads.")
        epsilon = float(attrs["epsilon"])
        rope_theta = float(attrs["rope_theta"])
        if epsilon <= 0 or rope_theta <= 0:
            raise IRSchemaError("Qwen3 attention epsilon and rope_theta must be positive.")
        attrs["epsilon"] = epsilon
        attrs["rope_theta"] = rope_theta
        return attrs

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value = cls.value.type_of(inputs)
        state = cls.state.type_of(inputs)
        assert isinstance(value, TensorType)
        hidden_size = value.shape[-1]
        query_size = int(attrs["num_attention_heads"]) * int(attrs["head_dim"])
        kv_size = int(attrs["num_key_value_heads"]) * int(attrs["head_dim"])
        q_weight = cls.q_weight.type_of(inputs)
        k_weight = cls.k_weight.type_of(inputs)
        v_weight = cls.v_weight.type_of(inputs)
        output_weight = cls.output_weight.type_of(inputs)
        hidden_extent = hidden_size.fixed_value
        if _fixed_shape(q_weight) != (query_size, hidden_extent):
            raise IRSchemaError("Qwen3 q_proj weight shape does not match attention dimensions.")
        if _fixed_shape(k_weight) != (kv_size, hidden_extent) or _fixed_shape(v_weight) != (kv_size, hidden_extent):
            raise IRSchemaError("Qwen3 k_proj/v_proj weight shape does not match attention dimensions.")
        if _fixed_shape(output_weight) != (hidden_extent, query_size):
            raise IRSchemaError("Qwen3 o_proj weight shape does not match attention dimensions.")
        if _fixed_shape(cls.q_norm_weight.type_of(inputs)) != (int(attrs["head_dim"]), ):
            raise IRSchemaError("Qwen3 q_norm weight must match head_dim.")
        if _fixed_shape(cls.k_norm_weight.type_of(inputs)) != (int(attrs["head_dim"]), ):
            raise IRSchemaError("Qwen3 k_norm weight must match head_dim.")
        return TupleType((value, state))

    @classmethod
    def infer_effect(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> Effect:
        return effect("read_write", "paged_attention_kv_cache")

    @classmethod
    def evaluate(cls, node, arguments, context):
        return qwen3_paged_attention(
            cls.value.read(arguments),
            cls.state.read(arguments),
            cls.q_weight.read(arguments),
            cls.k_weight.read(arguments),
            cls.v_weight.read(arguments),
            cls.q_norm_weight.read(arguments),
            cls.k_norm_weight.read(arguments),
            cls.output_weight.read(arguments),
            layer_id=_scalar_int(cls.layer_id.read(arguments)),
            num_attention_heads=int(node.attrs["num_attention_heads"]),
            num_key_value_heads=int(node.attrs["num_key_value_heads"]),
            head_dim=int(node.attrs["head_dim"]),
            epsilon=float(node.attrs["epsilon"]),
            rope_theta=float(node.attrs["rope_theta"]),
            advance_sequence=_scalar_bool(cls.advance_sequence.read(arguments)),
            torch=context.torch,
        )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(flops=None, bytes_read=None, bytes_written=None, notes=("stateful-qwen3-paged-attention",))


def qwen3_paged_attention(
    hidden,
    state,
    q_weight,
    k_weight,
    v_weight,
    q_norm_weight,
    k_norm_weight,
    output_weight,
    *,
    layer_id: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    epsilon: float,
    rope_theta: float,
    advance_sequence: bool = True,
    torch=None,
):
    if torch is None:
        import torch
    if not isinstance(state, PagedAttentionState):
        raise EvaluationError(f"Qwen3 attention state must be PagedAttentionState, got {type(state).__name__}.")
    state.validate()
    if hidden.shape[0] != 1:
        raise EvaluationError("Qwen3 P0 decode attention requires exactly one input token.")
    if (
        state.config.num_kv_heads != num_key_value_heads
        or state.config.head_dim != head_dim
        or state.config.num_layers <= layer_id
    ):
        raise EvaluationError("Paged-attention state config does not match the Qwen3 operation.")

    query = torch.nn.functional.linear(hidden, q_weight).reshape(num_attention_heads, head_dim)
    key = torch.nn.functional.linear(hidden, k_weight).reshape(num_key_value_heads, head_dim)
    value = torch.nn.functional.linear(hidden, v_weight).reshape(num_key_value_heads, head_dim)
    query = _rms_norm(query, q_norm_weight, epsilon)
    key = _rms_norm(key, k_norm_weight, epsilon)
    position = state.sequence_length
    query = _apply_rope(query, position=position, theta=rope_theta)
    key = _apply_rope(key, position=position, theta=rope_theta)
    state.append(
        key,
        value,
        layer_id=layer_id,
        advance_sequence=advance_sequence,
    )
    key_history, value_history = state.gather(
        layer_id=layer_id, length=position + 1
    )

    groups = num_attention_heads // num_key_value_heads
    head_to_kv = torch.arange(num_attention_heads, device=hidden.device) // groups
    expanded_key = key_history[:, head_to_kv, :].permute(1, 0, 2)
    expanded_value = value_history[:, head_to_kv, :].permute(1, 0, 2)
    scores = torch.einsum("hd,hsd->hs", query.float(), expanded_key.float()) * (head_dim ** -0.5)
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=query.dtype)
    attention = torch.einsum("hs,hsd->hd", probabilities, expanded_value)
    merged = attention.reshape(1, num_attention_heads * head_dim)
    return torch.nn.functional.linear(merged, output_weight), state


def _scalar_int(value) -> int:
    result = value.item() if hasattr(value, "item") else value
    if isinstance(result, bool) or not isinstance(result, int):
        raise EvaluationError(f"Qwen3 layer_id must evaluate to int32, got {result!r}.")
    return int(result)


def _scalar_bool(value) -> bool:
    result = value.item() if hasattr(value, "item") else value
    if not isinstance(result, bool):
        raise EvaluationError(
            f"Qwen3 advance_sequence must evaluate to bool, got {result!r}."
        )
    return result


def _rms_norm(value, weight, epsilon: float):
    normalized = value.float() * torch_rsqrt(value.float().pow(2).mean(dim=-1, keepdim=True) + epsilon)
    return normalized.to(dtype=value.dtype) * weight.to(dtype=value.dtype)


def torch_rsqrt(value):
    # Keeping this helper separate makes the exact cast boundary obvious in IR
    # diagnostics and avoids binding a module-global torch dependency.
    return value.rsqrt()


def _apply_rope(value, *, position: int, theta: float):
    torch = __import__("torch")
    head_dim = value.shape[-1]
    indices = torch.arange(0, head_dim, 2, dtype=torch.float32, device=value.device)
    inverse_frequency = theta ** (-indices / head_dim)
    angles = position * inverse_frequency
    frequency = torch.cat((angles, angles), dim=-1)
    cosine = frequency.cos().to(dtype=value.dtype)
    sine = frequency.sin().to(dtype=value.dtype)
    half = head_dim // 2
    rotated = torch.cat((-value[..., half:], value[..., :half]), dim=-1)
    return (value * cosine) + (rotated * sine)


def _fixed_shape(value: TensorType) -> tuple[int | None, ...]:
    return tuple(dimension.value for dimension in value.shape)


__all__ = ["Qwen3PagedAttention", "qwen3_paged_attention"]
