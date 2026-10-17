# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Sparse paged attention evaluation boundary for DSA top-k indices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.model import DistributedType, DType, Effect, IRType, Node, effect, tensor_type
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    ParameterKind,
    attribute_parameter,
    input_parameter,
    op_definition,
)
from triton.flagmega.ir.ops.nn._attention_layout import from_seq_head_dim, normalize_attention_layout, to_seq_head_dim
from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionState
from triton.flagmega.ir.ops.nn.dsa_indexer import _dsa_broadcast_choices
from triton.flagmega.ir.ops.nn.qwen3_paged_attention import _scalar_int
from triton.flagmega.ir.type_pattern import has_dtype, has_rank, is_ref, is_tensor
from triton.flagmega.ir.types import VectorType


@op_definition(
    "nn.dsa_paged_attention",
    namespace="nn",
    functional_name="dsa_paged_attention",
    display_name="NN.DSAPagedAttention",
)
class DsaPagedAttention(OpDefinition):
    q = input_parameter(is_tensor() & has_rank(3))
    topk = input_parameter(is_tensor() & has_rank(2) & has_dtype(DType.INT32))
    state = input_parameter(is_ref(), parameter_kind=ParameterKind.ATTRIBUTE,
                            memory_effect=MemoryEffect.CHIP_READ.partitioned_by_argument(3))
    layer_id = input_parameter(is_tensor() & has_rank(0) & has_dtype(DType.INT32),
                               parameter_kind=ParameterKind.ATTRIBUTE)
    scale = attribute_parameter()
    layout = attribute_parameter()
    hidden_size = attribute_parameter()
    lanes = attribute_parameter(default=8)
    probability_rounding = attribute_parameter(default="normalized")
    index_traversal = attribute_parameter(default="slots")

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        attrs["scale"] = float(attrs["scale"])
        if attrs["scale"] <= 0:
            raise IRSchemaError("DSA paged-attention scale must be positive.")
        attrs["layout"] = normalize_attention_layout(attrs["layout"])
        if isinstance(attrs["lanes"], bool) or not isinstance(attrs["lanes"], int) or attrs["lanes"] <= 0:
            raise IRSchemaError("DSA paged-attention lanes must be positive.")
        if isinstance(attrs["hidden_size"], bool) or not isinstance(attrs["hidden_size"], int) or attrs["hidden_size"] <= 0:
            raise IRSchemaError("DSA paged-attention hidden_size must be positive.")
        if attrs["probability_rounding"] not in {"normalized", "online_bf16"}:
            raise IRSchemaError("DSA probability_rounding must be normalized or online_bf16.")
        if not isinstance(attrs["index_traversal"], str) or attrs["index_traversal"] not in {"slots", "compact"}:
            raise IRSchemaError("DSA index_traversal must be slots or compact.")
        return attrs

    @classmethod
    def ir_attrs(cls, attrs):
        return {key: value for key, value in attrs.items()
                if not (key == "probability_rounding" and value == "normalized")
                and not (key == "index_traversal" and value == "slots")}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        query_type = cls.q.type_of(inputs)
        query = tensor_of(query_type)
        scalar = query.dtype.elem_type if isinstance(query.dtype, VectorType) else query.dtype
        if attrs.get("probability_rounding", "normalized") == "online_bf16" and scalar != DType.BFLOAT16:
            raise IRSchemaError("DSA online_bf16 probability_rounding requires BF16 queries.")
        layout = tuple(attrs["layout"])
        if not isinstance(query.dtype, VectorType):
            dim_axis = layout.index("dim")
            logical_dim = query.shape[dim_axis]
            lanes = int(attrs["lanes"])
            if logical_dim.is_fixed and logical_dim.fixed_value % lanes == 0:
                shape = list(query.shape)
                shape[dim_axis] = logical_dim.fixed_value // lanes
                packed = tensor_type(VectorType(query.dtype, (lanes,)), shape, layout=query.layout)
                query_type = (DistributedType(packed, query_type.axis_policies, query_type.placement,
                                               query_type.partial, query_type.exclusive)
                              if isinstance(query_type, DistributedType) else packed)
                query = packed
        if query.shape[layout.index("head")].is_fixed and query.shape[layout.index("dim")].is_fixed:
            lanes = query.dtype.lane_count if isinstance(query.dtype, VectorType) else 1
            if (query.shape[layout.index("head")].fixed_value
                    * query.shape[layout.index("dim")].fixed_value * lanes != attrs["hidden_size"]):
                raise IRSchemaError("DSA query heads/dim do not match hidden_size.")
        return query_type

    @classmethod
    def distributed_input_type_tuples(cls, choices, attrs):
        yield from _dsa_broadcast_choices(choices, (0, 1))

    @classmethod
    def infer_effect(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> Effect:
        return effect("read", "paged_attention_kv_cache")

    @classmethod
    def evaluate(cls, node, arguments, context):
        from triton.flagmega.ir.ops.ntt._paged_attention_split import pack_dim_from_scalar, unpack_dim_to_scalar
        input_type = tensor_of(context.types[cls.q.read(node.inputs)])
        query_type = input_type
        if not isinstance(input_type.dtype, VectorType):
            dim_axis = tuple(node.attrs["layout"]).index("dim")
            dim = input_type.shape[dim_axis].fixed_value
            lanes = int(node.attrs["lanes"])
            if dim % lanes:
                raise EvaluationError("DSA scalar query dimension is not divisible by the configured lanes.")
            shape = list(input_type.shape)
            shape[dim_axis] = dim // lanes
            query_type = tensor_type(VectorType(input_type.dtype, (lanes,)), shape,
                                     layout=input_type.layout)
        q = unpack_dim_to_scalar(cls.q.read(arguments), input_type, tuple(node.attrs["layout"]).index("dim"))
        q = to_seq_head_dim(q, tuple(node.attrs["layout"]))
        topk = cls.topk.read(arguments)
        state = cls.state.read(arguments)
        if not isinstance(state, PagedAttentionState):
            raise EvaluationError("DSA paged-attention state must evaluate to PagedAttentionState.")
        state.validate()
        layer = _scalar_int(cls.layer_id.read(arguments))
        base = int(state.slot_mapping[0].item())
        keys, values = state.gather(layer_id=layer, length=base + q.shape[0])
        results = []
        for row, query in enumerate(q):
            length = base + row + 1
            indices = topk[row].to(dtype=context.torch.int64)
            valid = (indices >= 0) & (indices < length)
            indices = indices[valid]
            if indices.numel() == 0:
                raise EvaluationError("DSA top-k selection produced no visible key.")
            if node.attrs.get("probability_rounding", "normalized") == "online_bf16":
                traversed = indices if node.attrs.get("index_traversal", "slots") == "compact" else topk[row]
                results.append(_online_attention(context.torch, query, traversed, keys, values,
                                                 length, float(node.attrs["scale"])))
                continue
            selected_keys = keys[indices]
            selected_values = values[indices]
            scores = context.torch.einsum("hd,shd->hs", query.float(), selected_keys.float()) * float(node.attrs["scale"])
            probabilities = context.torch.softmax(scores, dim=-1, dtype=context.torch.float32).to(dtype=query.dtype)
            results.append(context.torch.einsum("hs,shd->hd", probabilities, selected_values))
        result = context.torch.stack(results)
        result = from_seq_head_dim(result, tuple(node.attrs["layout"]))
        if isinstance(query_type.dtype, VectorType):
            result = pack_dim_from_scalar(result, query_type.dtype, tuple(node.attrs["layout"]).index("dim"))
        return result.contiguous()

    @classmethod
    def cost(cls, node):
        return OpCost(notes=("dsa-topk-paged-attention",))


__all__ = ["DsaPagedAttention"]


def _online_attention(torch, query, indices, keys, values, length, scale):
    """64-slot online softmax: cast exponentials before PV, normalize last."""
    maximum = torch.full((query.shape[0], 1), -float("inf"), dtype=torch.float32, device=query.device)
    denominator = torch.zeros_like(maximum)
    accumulator = torch.zeros_like(query, dtype=torch.float32)
    for start in range(0, indices.numel(), 64):
        selected = indices[start:start + 64].to(dtype=torch.int64)
        valid = (selected >= 0) & (selected < length)
        if not bool(valid.any()):
            continue
        selected = selected[valid]
        score = torch.einsum("hd,shd->hs", query.float(), keys[selected].float()) * scale
        updated = torch.maximum(maximum, score.max(-1, keepdim=True).values)
        previous_scale = torch.exp(maximum - updated)
        probability = torch.exp(score - updated)
        denominator = denominator * previous_scale + probability.sum(-1, keepdim=True)
        accumulator = accumulator * previous_scale + torch.einsum(
            "hs,shd->hd", probability.bfloat16().float(), values[selected].float())
        maximum = updated
    return (accumulator / denominator).to(dtype=query.dtype)
