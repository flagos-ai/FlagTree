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
    require_decode_token,
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

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        return normalize_paged_attention_attrs(attrs)

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        query_type = cls.q.type_of(inputs)
        query = tensor_of(query_type)
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
    return {
        "scale": scale,
        "layout": normalize_attention_layout(attributes["layout"]),
        "hidden_size": hidden_size,
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
):
    """Evaluate attention as scalar elements, independent of vector packing."""

    from triton.flagmega.ir.ops.ntt._paged_attention_split import (
        unpack_dim_to_scalar,
    )

    query_value = unpack_dim_to_scalar(
        query_value, query_type, layout.index("dim")
    )
    query_value = to_seq_head_dim(query_value, layout)
    query = require_decode_token(query_value, operation="PagedAttention")
    if not isinstance(state, PagedAttentionState):
        raise EvaluationError("PagedAttention state must evaluate to PagedAttentionState.")
    state.validate()
    layer_id = _scalar_int(layer_id_value)
    length = int(state.slot_mapping[0].item()) + 1
    key_history, value_history = state.gather(layer_id=layer_id, length=length)
    query_heads, head_dim = query.shape
    kv_heads = state.config.num_kv_heads
    if query_heads % kv_heads or head_dim != state.config.head_dim:
        raise EvaluationError("PagedAttention query shape disagrees with cache config.")
    groups = query_heads // kv_heads
    head_to_kv = context.torch.arange(query_heads, device=query.device) // groups
    expanded_key = key_history[:, head_to_kv, :].permute(1, 0, 2)
    expanded_value = value_history[:, head_to_kv, :].permute(1, 0, 2)
    scores = context.torch.einsum(
        "hd,hsd->hs", query.float(), expanded_key.float()
    ) * scale
    probabilities = context.torch.softmax(
        scores, dim=-1, dtype=context.torch.float32
    ).to(dtype=query.dtype)
    result = context.torch.einsum(
        "hs,hsd->hd", probabilities, expanded_value
    ).unsqueeze(0)
    return from_seq_head_dim(result, layout).contiguous()


__all__ = [
    "PagedAttention",
    "normalize_paged_attention_attrs",
    "paged_attention_scalar_value",
]
