# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Write a semantic K or V token chunk into a paged-attention cache."""

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
    normalize_attention_layout,
    to_seq_head_dim,
)
from triton.flagmega.ir.ops.nn._paged_attention_state import (
    PagedAttentionState,
    paged_attention_state_config_from_type,
)
from triton.flagmega.ir.ops.nn.qwen3_paged_attention import _scalar_bool, _scalar_int
from triton.flagmega.ir.ops.tensors.unpack import unpack_physical
from triton.flagmega.ir.type_pattern import has_dtype, has_rank, is_ref, is_tensor
from triton.flagmega.ir.types import VectorType


@op_definition(
    "nn.update_paged_attention_kv_cache",
    namespace="nn",
    functional_name="update_paged_attention_kv_cache",
    display_name="NN.UpdatePagedAttentionKVCache",
)
class UpdatePagedAttentionKVCache(OpDefinition):
    slots = input_parameter(is_tensor() & has_rank(3))
    state = input_parameter(
        is_ref(), parameter_kind=ParameterKind.ATTRIBUTE,
        memory_effect=MemoryEffect.CHIP_READ_WRITE.partitioned_by_argument(2),
    )
    layer_id = input_parameter(
        is_tensor() & has_rank(0) & has_dtype(DType.INT32),
        parameter_kind=ParameterKind.ATTRIBUTE,
    )
    advance_sequence = input_parameter(
        is_tensor() & has_rank(0) & has_dtype(DType.BOOL),
        parameter_kind=ParameterKind.ATTRIBUTE,
    )
    cache_kind = attribute_parameter()
    layout = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        cache_kind = str(attrs["cache_kind"]).lower()
        if cache_kind not in {"key", "value"}:
            raise IRSchemaError("UpdatePagedAttentionKVCache cache_kind must be key or value.")
        return {
            "cache_kind": cache_kind,
            "layout": normalize_attention_layout(attrs["layout"]),
        }

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        slots_type = cls.slots.type_of(inputs)
        # layer_id/advance_sequence are nncase-style Attribute operands.  They
        # remain scalar values at the launch ABI and do not participate in the
        # distributed type relation.
        if isinstance(slots_type, DistributedType):
            if slots_type.partial is not None:
                raise IRSchemaError(
                    "UpdatePagedAttentionKVCache requires materialized slots.")
            dim_axis = tuple(attrs["layout"]).index("dim")
            if not isinstance(slots_type.axis_policies[dim_axis], SBPBroadCast):
                raise IRSchemaError(
                    "UpdatePagedAttentionKVCache head-dimension axis cannot be split.")
        return cls.state.type_of(inputs)

    @classmethod
    def infer_effect(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> Effect:
        return effect("read_write", "paged_attention_kv_cache")

    @classmethod
    def evaluate(cls, node, arguments, context):
        slots = cls.slots.read(arguments)
        slots_type = tensor_of(context.types[cls.slots.read(node.inputs)])
        if isinstance(slots_type.dtype, VectorType):
            config = paged_attention_state_config_from_type(
                context.types[cls.state.read(node.inputs)]
            )
            if slots_type.dtype.lanes != (config.lanes,):
                raise EvaluationError(
                    "UpdatePagedAttentionKVCache slot vector lanes do not match "
                    "the configured cache."
                )
            slots = unpack_physical(
                slots,
                slots_type.rank,
                slots_type.dtype.lanes,
                (tuple(node.attrs["layout"]).index("dim"),),
            )
        slots = to_seq_head_dim(slots, tuple(node.attrs["layout"]))
        state = cls.state.read(arguments)
        if not isinstance(state, PagedAttentionState):
            raise EvaluationError(
                "UpdatePagedAttentionKVCache state must evaluate to PagedAttentionState.")
        state.update(
            slots,
            cache_kind=str(node.attrs["cache_kind"]),
            layer_id=_scalar_int(cls.layer_id.read(arguments)),
            advance_sequence=_scalar_bool(cls.advance_sequence.read(arguments)),
        )
        return state

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(notes=(f"paged-cache-{node.attrs['cache_kind']}-write",))


__all__ = ["UpdatePagedAttentionKVCache"]
