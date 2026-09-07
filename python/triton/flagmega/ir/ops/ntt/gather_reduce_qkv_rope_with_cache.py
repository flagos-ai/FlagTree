# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fused partial-QKV reduction, normalization, RoPE, and cache update."""

from __future__ import annotations

from dataclasses import replace
from math import prod
from types import SimpleNamespace
from typing import Mapping, Sequence

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import ReduceOp, SBPPartial
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.model import (
    DistributedType,
    DType,
    Effect,
    IRType,
    Node,
    TupleType,
)
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    ParameterKind,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_elements,
    tensor_nbytes,
)
from triton.flagmega.ir.ops.nn.qkv_rope_with_cache import QKVRoPEWithCache
from triton.flagmega.ir.ops.ntt.packed_qkv_parallel_linear_combine import (
    can_materialize_packed_qkv,
)
from triton.flagmega.ir.type_pattern import (
    has_dtype,
    has_rank,
    is_ref,
    is_tensor,
    is_tuple,
)
from triton.flagmega.ir.types import VectorType


@op_definition(
    "ntt.gather_reduce_qkv_rope_with_cache",
    namespace="ntt",
    functional_name="gather_reduce_qkv_rope_with_cache",
    display_name="NTT.GatherReduceQKVRoPEWithCache",
)
class GatherReduceQKVRoPEWithCache(OpDefinition):
    """Consume compact Sum-partial Q/K/V without materializing their sums.

    ``materialized_qkv_type`` records the collective result which was removed;
    ``logical_qkv_type`` records the three zero-copy Q/K/V views consumed by
    the original semantic operation.  Keeping both types in IR makes the
    transformation independently verifiable and preserves enough information
    for a backend to map logical coordinates back to partial-owner storage.
    """

    qkv = input_parameter(
        is_tuple(), memory_effect=MemoryEffect.READ.across_partial_owners()
    )
    q_scale = input_parameter(is_tensor())
    k_scale = input_parameter(is_tensor())
    q_bias = input_parameter(is_tensor())
    k_bias = input_parameter(is_tensor())
    cos = input_parameter(is_tensor())
    sin = input_parameter(is_tensor())
    state = input_parameter(
        is_ref(),
        parameter_kind=ParameterKind.ATTRIBUTE,
        memory_effect=MemoryEffect.CHIP_READ_WRITE.partitioned_by_argument(8),
    )
    layer_id = input_parameter(
        is_tensor() & has_rank(0) & has_dtype(DType.INT32),
        parameter_kind=ParameterKind.ATTRIBUTE,
    )
    advance_sequence = input_parameter(
        is_tensor() & has_rank(0) & has_dtype(DType.BOOL),
        parameter_kind=ParameterKind.ATTRIBUTE,
    )
    materialized_qkv_type = attribute_parameter()
    logical_qkv_type = attribute_parameter()
    q_axis = attribute_parameter()
    q_epsilon = attribute_parameter()
    q_use_mean = attribute_parameter()
    q_round_before_scale = attribute_parameter(default=False)
    k_axis = attribute_parameter()
    k_epsilon = attribute_parameter()
    k_use_mean = attribute_parameter()
    k_round_before_scale = attribute_parameter(default=False)
    qkv_layout = attribute_parameter()
    attention_layout = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        materialized = attrs["materialized_qkv_type"]
        logical = attrs["logical_qkv_type"]
        if not isinstance(materialized, TupleType) or not isinstance(logical, TupleType):
            raise IRSchemaError(
                "GatherReduceQKVRoPEWithCache materialized/logical QKV types "
                "must be TupleType values."
            )
        normalized = QKVRoPEWithCache.normalize_attrs(_base_qkv_attrs(attrs))
        return {
            "materialized_qkv_type": materialized,
            "logical_qkv_type": logical,
            **normalized,
        }

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        partial_type = cls.qkv.type_of(inputs)
        materialized_type = attrs["materialized_qkv_type"]
        logical_type = attrs["logical_qkv_type"]
        if not isinstance(partial_type, TupleType):
            raise IRSchemaError(
                "GatherReduceQKVRoPEWithCache requires a three-field partial tuple."
            )
        if not can_materialize_packed_qkv(partial_type, materialized_type):
            raise IRSchemaError(
                "GatherReduceQKVRoPEWithCache input cannot form its recorded "
                "materialized QKV type."
            )
        _validate_partial_and_view_types(partial_type, materialized_type, logical_type)
        logical_qkv = Node("__logical_qkv", "builtin.var", (), logical_type)
        return QKVRoPEWithCache.infer_type(
            (logical_qkv, *inputs[1:]), _base_qkv_attrs(attrs)
        )

    @classmethod
    def infer_effect(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> Effect:
        return QKVRoPEWithCache.infer_effect(inputs, _base_qkv_attrs(attrs))

    @classmethod
    def evaluate(cls, node, arguments, context):
        values = cls.qkv.read(arguments)
        logical_type = node.attrs["logical_qkv_type"]
        if (
            not isinstance(values, (tuple, list))
            or len(values) != 3
            or not isinstance(logical_type, TupleType)
        ):
            raise EvaluationError(
                "GatherReduceQKVRoPEWithCache expects three partial Q/K/V values."
            )
        logical_values = tuple(
            _reshape_physical_value(value, value_type)
            for value, value_type in zip(values, logical_type.fields, strict=True)
        )
        # Reference evaluation represents a distributed partial by its full
        # logical dense value.  Reshape that value through the recorded view
        # and reuse the semantic QKVRoPEWithCache evaluator; execution backends
        # perform the actual owner reduction described by this operation.
        proxy = replace(
            node,
            op=QKVRoPEWithCache.op_name,
            attrs=_base_qkv_attrs(node.attrs),
        )
        proxy_types = dict(context.types)
        proxy_types[node.inputs[0]] = logical_type
        proxy_context = SimpleNamespace(types=proxy_types, torch=context.torch)
        return QKVRoPEWithCache.evaluate(
            proxy, (logical_values, *arguments[1:]), proxy_context
        )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        output = node.type.fields[0] if isinstance(node.type, TupleType) else None
        elements = tensor_elements(output) if output is not None else None
        size = tensor_nbytes(output) if output is not None else None
        return OpCost(
            flops=None if elements is None else elements * 9,
            bytes_read=None if size is None else size * 4,
            bytes_written=size,
            communication_bytes=None,
            synchronizations=1,
            notes=("fused-sum-partial-qkv-normalization-rope-cache-update",),
        )


def _base_qkv_attrs(attrs: Mapping[str, object]) -> dict[str, object]:
    return {
        "q_round_before_scale": attrs.get("q_round_before_scale", False),
        "k_round_before_scale": attrs.get("k_round_before_scale", False),
        **{
        name: attrs[name]
        for name in (
            "q_axis",
            "q_epsilon",
            "q_use_mean",
            "k_axis",
            "k_epsilon",
            "k_use_mean",
            "qkv_layout",
            "attention_layout",
        )
        },
    }


def _validate_partial_and_view_types(
    partial: TupleType,
    materialized: TupleType,
    logical: TupleType,
) -> None:
    if not (len(partial.fields) == len(materialized.fields) == len(logical.fields) == 3):
        raise IRSchemaError(
            "GatherReduceQKVRoPEWithCache requires exactly three Q/K/V fields."
        )
    for label, source, combined, view in zip(
        ("Q", "K", "V"),
        partial.fields,
        materialized.fields,
        logical.fields,
        strict=True,
    ):
        if not all(
            isinstance(value, DistributedType) for value in (source, combined, view)
        ):
            raise IRSchemaError(
                f"GatherReduceQKVRoPEWithCache {label} types must be distributed."
            )
        assert isinstance(source, DistributedType)
        assert isinstance(combined, DistributedType)
        assert isinstance(view, DistributedType)
        if (
            source.partial is None
            or source.partial.reduce_op is not ReduceOp.SUM
            or not source.partial.axes
            or combined.partial is not None
            or view.partial is not None
            or any(isinstance(policy, SBPPartial) for policy in source.axis_policies)
            or any(isinstance(policy, SBPPartial) for policy in combined.axis_policies)
            or any(isinstance(policy, SBPPartial) for policy in view.axis_policies)
        ):
            raise IRSchemaError(
                f"GatherReduceQKVRoPEWithCache {label} requires a non-empty Sum "
                "partial input and materialized logical views."
            )
        if not (
            source.placement == combined.placement == view.placement
            and tensor_of(source).dtype == tensor_of(combined).dtype == tensor_of(view).dtype
            and tensor_elements(tensor_of(combined)) == tensor_elements(tensor_of(view))
        ):
            raise IRSchemaError(
                f"GatherReduceQKVRoPEWithCache {label} view changes placement, "
                "dtype, or scalar element count."
            )
        split_axes = {
            axis
            for policy in source.axis_policies
            if hasattr(policy, "hierarchy_axes")
            for axis in policy.hierarchy_axes
        }
        if split_axes & set(source.partial.axes):
            raise IRSchemaError(
                f"GatherReduceQKVRoPEWithCache {label} placement axes cannot be "
                "both split and partial."
            )


def _reshape_physical_value(value, value_type: IRType):
    tensor = tensor_of(value_type)
    dimensions = tuple(dimension.value for dimension in tensor.shape)
    if any(dimension is None for dimension in dimensions):
        raise EvaluationError(
            "GatherReduceQKVRoPEWithCache reference evaluation requires fixed views."
        )
    lanes = tensor.dtype.lanes if isinstance(tensor.dtype, VectorType) else ()
    shape = (*dimensions, *lanes)
    if int(value.numel()) != prod(shape, start=1):
        raise EvaluationError(
            "GatherReduceQKVRoPEWithCache partial value cannot be reshaped to "
            f"its logical view {shape}."
        )
    return value.reshape(shape)


__all__ = ["GatherReduceQKVRoPEWithCache"]
