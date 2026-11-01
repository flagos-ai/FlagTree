# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Target-independent paged attention over an explicitly updated cache."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import SBPBroadCast
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.model import DistributedType, DType, Effect, IRType, Node, effect
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    ParameterKind,
    attribute_parameter,
    input_parameter,
    op_definition,
)
from triton.flagmega.ir.ops.nn._attention_layout import (
    from_seq_head_dim,
    normalize_attention_layout,
    to_seq_head_dim,
)
from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionState
from triton.flagmega.ir.ops.nn.qwen3_paged_attention import _scalar_int
from triton.flagmega.ir.type_pattern import has_dtype, has_rank, is_ref, is_tensor
from triton.flagmega.ir.types import VectorType


@op_definition(
    "nn.paged_attention",
    namespace="nn",
    functional_name="paged_attention",
    display_name="NN.PagedAttention",
)
class PagedAttention(OpDefinition):
    q = input_parameter(is_tensor() & has_rank(3))
    state = input_parameter(
        is_ref(), parameter_kind=ParameterKind.ATTRIBUTE,
        memory_effect=MemoryEffect.CHIP_READ.partitioned_by_argument(2),
    )
    layer_id = input_parameter(
        is_tensor() & has_rank(0) & has_dtype(DType.INT32),
        parameter_kind=ParameterKind.ATTRIBUTE,
    )
    scale = attribute_parameter()
    layout = attribute_parameter()
    hidden_size = attribute_parameter()
    probability_rounding = attribute_parameter(default="normalized")
    softmax_block_size = attribute_parameter(default=64)
    softmax_traversal = attribute_parameter(default="forward")

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        return normalize_paged_attention_attrs(attrs)

    @classmethod
    def ir_attrs(cls, attrs):
        return {key: value for key, value in attrs.items()
                if not (attrs.get("probability_rounding", "normalized") == "normalized"
                        and key in {"probability_rounding", "softmax_block_size", "softmax_traversal"})}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        query_type = cls.q.type_of(inputs)
        query = tensor_of(query_type)
        scalar = query.dtype.elem_type if isinstance(query.dtype, VectorType) else query.dtype
        if attrs.get("probability_rounding", "normalized") == "online_bf16" and scalar != DType.BFLOAT16:
            raise IRSchemaError("PagedAttention online BF16 probabilities require BF16 queries.")
        layout = tuple(attrs["layout"])
        head_axis = layout.index("head")
        dim_axis = layout.index("dim")
        if (
            query.shape[head_axis].is_fixed
            and query.shape[dim_axis].is_fixed
            and query.shape[head_axis].fixed_value
            * query.shape[dim_axis].fixed_value
            * (query.dtype.lane_count if isinstance(query.dtype, VectorType) else 1)
            != int(attrs["hidden_size"])
        ):
            raise IRSchemaError(
                "PagedAttention query heads/dim do not match hidden_size.")
        if isinstance(query_type, DistributedType):
            if query_type.partial is not None:
                raise IRSchemaError("PagedAttention requires a materialized query.")
            if not isinstance(query_type.axis_policies[dim_axis], SBPBroadCast):
                raise IRSchemaError(
                    "PagedAttention head-dimension axis cannot be split.")
        return query_type

    @classmethod
    def infer_effect(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> Effect:
        return effect("read", "paged_attention_kv_cache")

    @classmethod
    def evaluate(cls, node, arguments, context):
        query_type = tensor_of(context.types[cls.q.read(node.inputs)])
        result = paged_attention_scalar_value(
            cls.q.read(arguments),
            query_type,
            cls.state.read(arguments),
            cls.layer_id.read(arguments),
            scale=float(node.attrs["scale"]),
            layout=tuple(node.attrs["layout"]),
            context=context,
            probability_rounding=node.attrs.get("probability_rounding", "normalized"),
            softmax_block_size=node.attrs.get("softmax_block_size", 64),
            softmax_traversal=node.attrs.get("softmax_traversal", "forward"),
        )
        if isinstance(query_type.dtype, VectorType):
            from triton.flagmega.ir.ops.ntt._paged_attention_split import (
                pack_dim_from_scalar,
            )

            result = pack_dim_from_scalar(
                result,
                query_type.dtype,
                tuple(node.attrs["layout"]).index("dim"),
            )
        return result.contiguous()

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(notes=("stateful-paged-attention-read",))


def normalize_paged_attention_attrs(
    attributes: Mapping[str, object],
) -> dict[str, object]:
    scale = float(attributes["scale"])
    hidden_size = attributes["hidden_size"]
    if scale <= 0:
        raise IRSchemaError("PagedAttention scale must be positive.")
    if (
        isinstance(hidden_size, bool)
        or not isinstance(hidden_size, int)
        or hidden_size <= 0
    ):
        raise IRSchemaError("PagedAttention hidden_size must be a positive integer.")
    rounding = attributes.get("probability_rounding", "normalized")
    block = attributes.get("softmax_block_size", 64)
    if rounding not in {"normalized", "online_bf16"}:
        raise IRSchemaError("PagedAttention probability_rounding must be normalized or online_bf16.")
    if isinstance(block, bool) or not isinstance(block, int) or block < 16 or block % 16:
        raise IRSchemaError("PagedAttention softmax_block_size must be a positive multiple of 16.")
    traversal = attributes.get("softmax_traversal", "forward")
    if traversal not in {"forward", "reverse"}:
        raise IRSchemaError("PagedAttention softmax_traversal must be forward or reverse.")
    return {
        "scale": scale,
        "layout": normalize_attention_layout(attributes["layout"]),
        "hidden_size": hidden_size,
        **({"probability_rounding": rounding, "softmax_block_size": block, "softmax_traversal": traversal}
           if rounding != "normalized" else {}),
    }


def paged_attention_scalar_value(
    query_value,
    query_type,
    state,
    layer_id_value,
    *,
    scale: float,
    layout: tuple[str, str, str],
    context,
    probability_rounding="normalized",
    softmax_block_size=64,
    softmax_traversal="forward",
):
    """Evaluate attention as scalar elements, independent of vector packing."""

    from triton.flagmega.ir.ops.ntt._paged_attention_split import (
        unpack_dim_to_scalar,
    )

    query_value = unpack_dim_to_scalar(
        query_value, query_type, layout.index("dim")
    )
    query_value = to_seq_head_dim(query_value, layout)
    if query_value.ndim != 3 or query_value.shape[0] == 0:
        raise EvaluationError("PagedAttention requires a nonempty [seq, head, dim] query chunk.")
    if not isinstance(state, PagedAttentionState):
        raise EvaluationError("PagedAttention state must evaluate to PagedAttentionState.")
    state.validate()
    layer_id = _scalar_int(layer_id_value)
    base_position = int(state.slot_mapping[0].item())
    if base_position < 0:
        raise EvaluationError("PagedAttention query chunk base must be non-negative.")
    tokens, query_heads, head_dim = query_value.shape
    key_history, value_history = state.gather(layer_id=layer_id, length=base_position + tokens)
    kv_heads = state.config.num_kv_heads
    if query_heads % kv_heads or head_dim != state.config.head_dim:
        raise EvaluationError("PagedAttention query shape disagrees with cache config.")
    groups = query_heads // kv_heads
    head_to_kv = context.torch.arange(query_heads, device=query_value.device) // groups
    expanded_key = key_history[:, head_to_kv, :].permute(1, 0, 2)
    expanded_value = value_history[:, head_to_kv, :].permute(1, 0, 2)
    results = []
    for row, query in enumerate(query_value):
        length = base_position + row + 1
        if probability_rounding == "online_bf16":
            torch = context.torch
            maximum = torch.full((query_heads, 1), -float("inf"), device=query.device, dtype=torch.float32)
            denominator = torch.zeros_like(maximum)
            accumulator = torch.zeros_like(query, dtype=torch.float32)
            starts = range(0, length, softmax_block_size)
            if softmax_traversal == "reverse":
                starts = reversed(starts)
            for start in starts:
                keys = expanded_key[:, start:min(start + softmax_block_size, length)]
                values = expanded_value[:, start:min(start + softmax_block_size, length)]
                valid = keys.shape[1]
                if query.is_cuda and valid < softmax_block_size:
                    # Retain the declared MMA tile for a causal tail. A short
                    # GEMM can select a different accumulation schedule even
                    # when all products have the same BF16 operand dtype.
                    padding = (0, 0, 0, softmax_block_size - valid)
                    keys = torch.nn.functional.pad(keys, padding)
                    values = torch.nn.functional.pad(values, padding)
                score = (torch.bmm(query[:, None, :], keys.transpose(1, 2), out_dtype=torch.float32).squeeze(1)
                         if query.is_cuda else torch.einsum("hd,hsd->hs", query.float(), keys.float())) * scale
                if query.is_cuda and valid < softmax_block_size:
                    score = score.masked_fill(torch.arange(softmax_block_size, device=query.device)[None, :] >= valid,
                                              -float("inf"))
                updated = torch.maximum(maximum, score.max(-1, keepdim=True).values)
                previous_scale = torch.exp2((maximum - updated) * 1.4426950408889634)
                if query.is_cuda:
                    from triton.flagmega.evaluator.cuda_attention_math import probability as online_probability
                    probability = online_probability(score, updated)
                else:
                    probability = torch.exp2(score * 1.4426950408889634 - updated * 1.4426950408889634)
                denominator = denominator * previous_scale + probability.sum(-1, keepdim=True)
                accumulator = accumulator * previous_scale
                if query.is_cuda:
                    accumulator = torch.baddbmm(accumulator[:, None, :], probability.bfloat16()[:, None, :],
                                               values, out_dtype=torch.float32).squeeze(1)
                else:
                    accumulator += torch.einsum("hs,hsd->hd", probability.bfloat16().float(), values.float())
                maximum = updated
            if query.is_cuda:
                from triton.flagmega.evaluator.cuda_attention_math import normalize
                result = normalize(accumulator, denominator)
            else:
                result = accumulator / denominator
            results.append(result.to(query.dtype))
            continue
        scores = context.torch.einsum(
            "hd,hsd->hs", query.float(), expanded_key[:, :length].float()
        ) * scale
        probabilities = context.torch.softmax(
            scores, dim=-1, dtype=context.torch.float32
        ).to(dtype=query.dtype)
        results.append(context.torch.einsum("hs,hsd->hd", probabilities, expanded_value[:, :length]))
    result = context.torch.stack(results)
    return from_seq_head_dim(result, layout).contiguous()


__all__ = [
    "PagedAttention",
    "normalize_paged_attention_attrs",
    "paged_attention_scalar_value",
]
