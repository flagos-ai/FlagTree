# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Portable implementation lookup for fused Q/K norm, RoPE, and cache IO."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

from triton.flagmega.errors import CodegenError

from .core import TIRMicroKernelContext, TIRMicroKernelProposal


class QKVRoPEWithCacheMicroKernelProvider:
    """Map target-neutral fused semantics through an injected target catalog."""

    op_names = frozenset({"nn.qkv_rope_with_cache"})
    family = "qkv_rope_with_cache"

    def propose(
        self, context: TIRMicroKernelContext
    ) -> TIRMicroKernelProposal | None:
        dispatch = context.dispatch
        if dispatch.semantic_op not in self.op_names:
            return None
        if len(dispatch.arguments) != 10 or len(dispatch.outputs) != 2:
            raise CodegenError(
                "QKVRoPEWithCache semantic TIR requires 10 arguments and 2 "
                f"outputs, got {len(dispatch.arguments)} and "
                f"{len(dispatch.outputs)}."
            )
        _validate_contract(dispatch.semantic_attrs)
        implementations = context.implementations(self.family, mode="decode")
        if not implementations:
            raise CodegenError(
                f"Implementation model {context.implementation_model.name!r} has "
                "no decode implementation for QKVRoPEWithCache."
            )
        candidates = tuple(context.candidate(value) for value in implementations)
        return TIRMicroKernelProposal(
            candidates,
            context.choose_default(self.family, candidates),
        )


def _validate_contract(attrs: Mapping[str, object]) -> None:
    qkv_layout = _layout(attrs, "qkv_layout")
    _layout(attrs, "attention_layout")
    rank = len(qkv_layout)
    dimension_axis = qkv_layout.index("dim")
    for prefix in ("q", "k"):
        axis = attrs.get(f"{prefix}_axis")
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise CodegenError(
                f"QKVRoPEWithCache {prefix}_axis must be an integer."
            )
        normalized = axis + rank if axis < 0 else axis
        if normalized < 0 or normalized >= rank or normalized > dimension_axis:
            raise CodegenError(
                f"QKVRoPEWithCache {prefix}_axis {axis} cannot normalize the "
                f"dimension axis {dimension_axis}."
            )
        epsilon = attrs.get(f"{prefix}_epsilon")
        if isinstance(epsilon, bool) or not isinstance(epsilon, Real) or epsilon <= 0:
            raise CodegenError(
                f"QKVRoPEWithCache {prefix}_epsilon must be positive."
            )
        if not isinstance(attrs.get(f"{prefix}_use_mean"), bool):
            raise CodegenError(
                f"QKVRoPEWithCache {prefix}_use_mean must be boolean."
            )


def _layout(attrs: Mapping[str, object], name: str) -> tuple[str, str, str]:
    raw = attrs.get(name)
    if not isinstance(raw, (tuple, list)):
        raise CodegenError(f"QKVRoPEWithCache requires {name} layout metadata.")
    value = tuple(str(axis) for axis in raw)
    if len(value) != 3 or set(value) != {"seq", "head", "dim"}:
        raise CodegenError(
            f"QKVRoPEWithCache has invalid {name} layout {value!r}."
        )
    return value


__all__ = ["QKVRoPEWithCacheMicroKernelProvider"]
