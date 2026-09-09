# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Portable microkernel contracts for decomposed attention primitives."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

from triton.flagmega.errors import CodegenError, IRSchemaError
from triton.flagmega.ir import get_definition

from .core import TIRMicroKernelContext, TIRMicroKernelProposal


_FAMILY_BY_OP = {
    "nn.rotary_embedding": "rotary_embedding",
    "nn.rope": "rope",
    "nn.update_paged_attention_kv_cache": "update_paged_attention_kv_cache",
    "ntt.vectorized_rope": "rope",
}

_ARITY_BY_OP = {
    "nn.rotary_embedding": (2, 2),
    "nn.rope": (3, 1),
    "nn.update_paged_attention_kv_cache": (4, 1),
    "ntt.vectorized_rope": (3, 1),
}


class AttentionPrimitiveMicroKernelProvider:
    """Resolve attention semantics through an injected implementation model."""

    op_names = frozenset(_FAMILY_BY_OP)

    def propose(
        self, context: TIRMicroKernelContext
    ) -> TIRMicroKernelProposal | None:
        dispatch = context.dispatch
        operation = dispatch.semantic_op
        family = _FAMILY_BY_OP.get(operation)
        if family is None:
            return None
        _validate_arity(
            operation,
            dispatch.arguments,
            dispatch.outputs,
        )
        _validate_semantic_contract(operation, dispatch.semantic_attrs)
        implementations = context.implementations(family, mode="decode")
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


def _validate_arity(
    operation: str,
    arguments: tuple[str, ...],
    outputs: tuple[str, ...],
) -> None:
    expected_arguments, expected_outputs = _ARITY_BY_OP[operation]
    if len(arguments) != expected_arguments or len(outputs) != expected_outputs:
        raise CodegenError(
            f"Semantic op {operation!r} requires {expected_arguments} arguments and "
            f"{expected_outputs} outputs, got {len(arguments)} and {len(outputs)}."
        )


def _validate_semantic_contract(
    operation: str, attrs: Mapping[str, object]
) -> None:
    if operation == "nn.rotary_embedding":
        head_dim = _positive_int(attrs, "head_dim", operation)
        if head_dim % 2:
            raise CodegenError("RotaryEmbedding head_dim must be even.")
        _positive_real(attrs, "theta", operation)
        _positive_real(attrs, "attention_scaling", operation)
        return
    if operation in {"nn.rope", "ntt.vectorized_rope"}:
        try:
            get_definition(operation).normalize_attrs(attrs)
        except IRSchemaError as error:
            raise CodegenError(str(error)) from error
        return
    if operation == "nn.update_paged_attention_kv_cache":
        if attrs.get("cache_kind") not in {"key", "value"}:
            raise CodegenError("Paged-attention cache_kind must be key or value.")
        _attention_layout(attrs, operation)
        return
    raise CodegenError(f"Unknown attention semantic op {operation!r}.")


def _attention_layout(
    attrs: Mapping[str, object], operation: str
) -> tuple[str, str, str]:
    value = attrs.get("layout")
    if not isinstance(value, (tuple, list)):
        raise CodegenError(f"Semantic op {operation!r} requires an attention layout.")
    layout = tuple(str(axis) for axis in value)
    if len(layout) != 3 or set(layout) != {"seq", "head", "dim"}:
        raise CodegenError(
            f"Semantic op {operation!r} has invalid attention layout {layout!r}."
        )
    return layout


def _positive_int(
    attrs: Mapping[str, object], name: str, operation: str
) -> int:
    value = attrs.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CodegenError(
            f"Semantic op {operation!r} requires positive integer {name}."
        )
    return value


def _positive_real(
    attrs: Mapping[str, object], name: str, operation: str
) -> float:
    value = attrs.get(name)
    if isinstance(value, bool) or not isinstance(value, Real) or value <= 0:
        raise CodegenError(
            f"Semantic op {operation!r} requires positive real {name}."
        )
    return float(value)


__all__ = ["AttentionPrimitiveMicroKernelProvider"]
