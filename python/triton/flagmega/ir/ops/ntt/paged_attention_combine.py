# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Merge explicit paged-attention online-softmax states."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.model import IRType, Node
from triton.flagmega.ir.ops.core import (
    CostKind,
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.ops.nn._attention_layout import normalize_attention_layout
from triton.flagmega.ir.ops.ntt._paged_attention_split import (
    create_combine_output_type,
    pack_dim_from_scalar,
)
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import (
    DType,
    VectorType,
    data_type,
    data_type_to_data,
)


@op_definition(
    "ntt.paged_attention_combine",
    namespace="ntt",
    functional_name="paged_attention_combine",
    display_name="NTT.PagedAttentionCombine",
)
class PagedAttentionCombine(OpDefinition):
    max_state = input_parameter(
        is_tensor(), memory_effect=MemoryEffect.READ.across_partial_owners()
    )
    sum_state = input_parameter(
        is_tensor(), memory_effect=MemoryEffect.READ.across_partial_owners()
    )
    acc_state = input_parameter(
        is_tensor(), memory_effect=MemoryEffect.READ.across_partial_owners()
    )
    layout = attribute_parameter()
    hidden_size = attribute_parameter()
    output_data_type = attribute_parameter()
    output_type = attribute_parameter()
    split_hierarchy_axis = attribute_parameter()
    split_count = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        hidden_size = attrs["hidden_size"]
        axis = attrs["split_hierarchy_axis"]
        count = attrs["split_count"]
        if (
            isinstance(hidden_size, bool)
            or not isinstance(hidden_size, int)
            or hidden_size <= 0
        ):
            raise IRSchemaError(
                "PagedAttentionCombine hidden_size must be a positive integer."
            )
        if isinstance(axis, bool) or not isinstance(axis, int) or axis < 0:
            raise IRSchemaError(
                "PagedAttentionCombine split_hierarchy_axis must be non-negative."
            )
        if isinstance(count, bool) or not isinstance(count, int) or count <= 1:
            raise IRSchemaError("PagedAttentionCombine split_count must exceed one.")
        output_dtype = data_type(attrs["output_data_type"])
        if not isinstance(output_dtype, (DType, VectorType)):
            raise IRSchemaError(
                "PagedAttentionCombine output_data_type must be scalar or vector."
            )
        if not isinstance(attrs["output_type"], IRType):
            raise IRSchemaError("PagedAttentionCombine output_type must be an IRType.")
        return {
            "layout": normalize_attention_layout(attrs["layout"]),
            "hidden_size": hidden_size,
            "output_data_type": output_dtype,
            "output_type": attrs["output_type"],
            "split_hierarchy_axis": axis,
            "split_count": count,
        }

    @classmethod
    def ir_attrs(cls, attrs: Mapping[str, object]) -> Mapping[str, object]:
        return {
            **dict(attrs),
            "output_data_type": data_type_to_data(data_type(attrs["output_data_type"])),
        }

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        return create_combine_output_type(
            cls.max_state.type_of(inputs),
            cls.sum_state.type_of(inputs),
            cls.acc_state.type_of(inputs),
            attrs["layout"],
            int(attrs["hidden_size"]),
            data_type(attrs["output_data_type"]),
            cls.output_type.read(inputs, attrs),
            int(attrs["split_hierarchy_axis"]),
            int(attrs["split_count"]),
        )

    @classmethod
    def evaluate(cls, node, arguments, context):
        # Max participates in the real online-softmax collective. The dense
        # evaluator receives already-normalized partial reference states, like
        # nncase's evaluator, so only acc/sum remains here.
        _ = cls.max_state.read(arguments)
        dtype = data_type(node.attrs["output_data_type"])
        scalar_dtype = dtype.elem_type if isinstance(dtype, VectorType) else dtype
        output = (
            cls.acc_state.read(arguments) / cls.sum_state.read(arguments)
        ).to(dtype=context.torch_dtype(scalar_dtype))
        return pack_dim_from_scalar(
            output, dtype, tuple(node.attrs["layout"]).index("dim")
        )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        output_bytes = tensor_nbytes(tensor_of(node.type))
        return OpCost(
            bytes_read=None,
            bytes_written=output_bytes,
            communication_bytes=None,
            synchronizations=None,
            kind=CostKind.ANALYTIC,
            model="flagmega.paged-attention-combine/v1",
            notes=("online-softmax-state-materialization",),
        )


__all__ = ["PagedAttentionCombine"]
