# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Portable implementation lookup for split paged-attention semantic TIR."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import (
    DType,
    DistributedType,
    IRType,
    RefType,
    TensorType,
    VectorType,
    logical_type,
)
from triton.flagmega.ir.distributed_type import local_shape

from .core import TIRMicroKernelContext, TIRMicroKernelProposal


_FAMILY_BY_OP = {
    "ntt.paged_attention_partial": "paged_attention_partial",
    "ntt.paged_attention_combine": "paged_attention_combine",
}

_ARITY_BY_OP = {
    "ntt.paged_attention_partial": (3, 3),
    "ntt.paged_attention_combine": (3, 1),
}


class PagedAttentionSplitMicroKernelProvider:
    """Map target-neutral split semantics through an injected target catalog."""

    op_names = frozenset(_FAMILY_BY_OP)

    def propose(
        self, context: TIRMicroKernelContext
    ) -> TIRMicroKernelProposal | None:
        dispatch = context.dispatch
        operation = dispatch.semantic_op
        family = _FAMILY_BY_OP.get(operation)
        if family is None:
            return None
        expected_inputs, expected_outputs = _ARITY_BY_OP[operation]
        if (
            len(dispatch.arguments) != expected_inputs
            or len(dispatch.outputs) != expected_outputs
        ):
            raise CodegenError(
                f"Semantic op {operation!r} requires {expected_inputs} arguments "
                f"and {expected_outputs} outputs, got {len(dispatch.arguments)} "
                f"and {len(dispatch.outputs)}."
            )
        _validate_contract(operation, dispatch.semantic_attrs)
        implementations = tuple(
            implementation
            for implementation in context.implementations(family, mode="decode")
            if _applicable(context, implementation.contract, implementation.parameters)
        )
        if not implementations:
            raise CodegenError(
                f"Implementation model {context.implementation_model.name!r} has no "
                f"decode implementation for semantic op {operation!r}."
            )
        candidates = tuple(context.candidate(value) for value in implementations)
        return TIRMicroKernelProposal(
            candidates,
            context.choose_default(family, candidates),
        )


def _applicable(context, contract, parameters) -> bool:
    """Match optional machine profiles to the semantic owner-local ABI."""

    if context.dispatch.semantic_op != "ntt.paged_attention_partial":
        return True
    constrained = any(str(key).startswith("required_") for key in contract)
    if not constrained:
        return True
    dispatch = context.dispatch
    parameter_map = context.function.parameter_map
    try:
        query_type = parameter_map[dispatch.arguments[0]].type
        state_type = logical_type(parameter_map[dispatch.arguments[1]].type)
    except (IndexError, KeyError):
        return False
    query = logical_type(query_type)
    if (
        not isinstance(query, TensorType)
        or query.rank != 3
        or not isinstance(state_type, RefType)
    ):
        return False
    fields = dict(state_type.fields)
    cache = fields.get("kv_caches")
    if not isinstance(cache, TensorType) or cache.rank != 6:
        return False
    layout = tuple(str(value) for value in dispatch.semantic_attrs.get("layout", ()))
    if len(layout) != 3 or set(layout) != {"seq", "head", "dim"}:
        return False
    query_shape = _fixed_local_shape(query_type)
    cache_shape = _fixed_shape(cache)
    if query_shape is None or cache_shape is None:
        return False
    head_axis = layout.index("head")
    dim_axis = layout.index("dim")
    query_lanes = _lanes(query.dtype)
    cache_lanes = _lanes(cache.dtype)
    query_head_dim = query_shape[dim_axis] * _lane_count(query.dtype)
    cache_head_dim = cache_shape[-1] * _lane_count(cache.dtype)
    observed = {
        "required_query_dtype": _scalar_dtype(query.dtype).value,
        "required_cache_dtype": _scalar_dtype(cache.dtype).value,
        "required_query_lanes": query_lanes,
        "required_cache_lanes": cache_lanes,
        "required_local_query_tokens": query_shape[layout.index("seq")],
        "required_local_query_heads": query_shape[head_axis],
        "required_scalar_head_dim": query_head_dim,
    }
    if any(
        contract.get(key) is not None and contract[key] != value
        for key, value in observed.items()
    ):
        return False
    block_n = int(parameters.get("block_n", 1))
    block_k = int(parameters.get("block_k", query_head_dim))
    return (
        query_head_dim == cache_head_dim == block_k
        and cache_shape[3] % block_n == 0
        and cache_shape[-2] > 0
    )


def _fixed_shape(value: TensorType) -> tuple[int, ...] | None:
    if any(not dimension.is_fixed for dimension in value.shape):
        return None
    return tuple(dimension.fixed_value for dimension in value.shape)


def _fixed_local_shape(value) -> tuple[int, ...] | None:
    tensor = logical_type(value)
    if not isinstance(tensor, TensorType):
        return None
    shape = local_shape(value) if isinstance(value, DistributedType) else tensor.shape
    if any(not dimension.is_fixed for dimension in shape):
        return None
    return tuple(dimension.fixed_value for dimension in shape)


def _scalar_dtype(value) -> DType:
    return value.elem_type if isinstance(value, VectorType) else value


def _lanes(value) -> tuple[int, ...]:
    return tuple(value.lanes) if isinstance(value, VectorType) else ()


def _lane_count(value) -> int:
    result = 1
    for lane in _lanes(value):
        result *= lane
    return result


def _validate_contract(operation: str, attrs: Mapping[str, object]) -> None:
    layout = attrs.get("layout")
    if not isinstance(layout, (tuple, list)) or set(layout) != {
        "seq", "head", "dim"
    } or len(layout) != 3:
        raise CodegenError(
            f"Semantic op {operation!r} requires one seq/head/dim layout."
        )
    _positive_int(attrs, "hidden_size", operation)
    split_axis = attrs.get("split_hierarchy_axis")
    if isinstance(split_axis, bool) or not isinstance(split_axis, int) or split_axis < 0:
        raise CodegenError(
            f"Semantic op {operation!r} requires a non-negative split_hierarchy_axis."
        )
    _positive_int(attrs, "split_count", operation, minimum=2)
    if operation == "ntt.paged_attention_partial":
        scale = attrs.get("scale")
        if isinstance(scale, bool) or not isinstance(scale, Real) or scale <= 0:
            raise CodegenError(
                "PagedAttentionPartial requires a positive attention scale."
            )
        return
    if not isinstance(attrs.get("output_type"), IRType):
        raise CodegenError("PagedAttentionCombine requires a typed output contract.")
    if "output_data_type" not in attrs:
        raise CodegenError("PagedAttentionCombine requires output_data_type.")


def _positive_int(
    attrs: Mapping[str, object],
    name: str,
    operation: str,
    *,
    minimum: int = 1,
) -> int:
    value = attrs.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CodegenError(
            f"Semantic op {operation!r} requires {name} >= {minimum}."
        )
    return value


__all__ = ["PagedAttentionSplitMicroKernelProvider"]
