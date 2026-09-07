# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit per-partition online-softmax states for paged attention."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.model import DType, Effect, IRType, Node, TupleType
from triton.flagmega.ir.ops.core import (
    CostKind,
    OpCost,
    OpDefinition,
    ParameterKind,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.ops.nn.paged_attention import (
    PagedAttention,
    normalize_paged_attention_attrs,
    paged_attention_scalar_value,
)
from triton.flagmega.ir.ops.ntt._paged_attention_split import (
    create_partial_state_type,
)
from triton.flagmega.ir.type_pattern import has_dtype, has_rank, is_ref, is_tensor


@op_definition(
    "ntt.paged_attention_partial",
    namespace="ntt",
    functional_name="paged_attention_partial",
    display_name="NTT.PagedAttentionPartial",
)
class PagedAttentionPartial(OpDefinition):
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
    split_hierarchy_axis = attribute_parameter()
    split_count = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        result = normalize_paged_attention_attrs(attrs)
        axis = attrs["split_hierarchy_axis"]
        count = attrs["split_count"]
        if isinstance(axis, bool) or not isinstance(axis, int) or axis < 0:
            raise IRSchemaError(
                "PagedAttentionPartial split_hierarchy_axis must be non-negative."
            )
        if isinstance(count, bool) or not isinstance(count, int) or count <= 1:
            raise IRSchemaError("PagedAttentionPartial split_count must exceed one.")
        result.update({
            "split_hierarchy_axis": axis,
            "split_count": count,
        })
        return result

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        attention_type = PagedAttention.infer_type(inputs, attrs)
        return create_partial_state_type(
            attention_type,
            attrs["layout"],
            int(attrs["hidden_size"]),
            int(attrs["split_hierarchy_axis"]),
            int(attrs["split_count"]),
        )

    @classmethod
    def infer_effect(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> Effect:
        return PagedAttention.infer_effect(inputs, attrs)

    @classmethod
    def evaluate(cls, node, arguments, context):
        attention = paged_attention_scalar_value(
            cls.q.read(arguments),
            tensor_of(context.types[cls.q.read(node.inputs)]),
            cls.state.read(arguments),
            cls.layer_id.read(arguments),
            scale=float(node.attrs["scale"]),
            layout=tuple(node.attrs["layout"]),
            context=context,
        ).float()
        dim_axis = tuple(node.attrs["layout"]).index("dim")
        stats_shape = list(attention.shape)
        stats_shape[dim_axis] = 1
        return (
            context.torch.zeros(
                stats_shape, dtype=context.torch.float32, device=attention.device
            ),
            context.torch.ones(
                stats_shape, dtype=context.torch.float32, device=attention.device
            ),
            attention,
        )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        if not isinstance(node.type, TupleType):
            return OpCost(notes=("invalid-paged-attention-partial-type",))
        writes = tuple(tensor_nbytes(tensor_of(field)) for field in node.type.fields)
        bytes_written = None if any(value is None for value in writes) else sum(
            value for value in writes if value is not None
        )
        return OpCost(
            bytes_read=None,
            bytes_written=bytes_written,
            communication_bytes=None,
            synchronizations=None,
            kind=CostKind.ANALYTIC,
            model="flagmega.paged-attention-partial/v1",
            notes=("split-kv-online-softmax-state",),
        )


__all__ = ["PagedAttentionPartial"]
