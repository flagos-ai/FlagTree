# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Microkernel selection for fused partial-QKV reduction and RoPE/cache IO."""

from __future__ import annotations

from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import DistributedType, TupleType, type_from_data

from .core import TIRMicroKernelContext, TIRMicroKernelProposal
from .qkv_rope_with_cache import _validate_contract


class GatherReduceQKVRoPEWithCacheMicroKernelProvider:
    op_names = frozenset({"ntt.gather_reduce_qkv_rope_with_cache"})
    family = "gather_reduce_qkv_rope_with_cache"

    def propose(
        self, context: TIRMicroKernelContext
    ) -> TIRMicroKernelProposal | None:
        dispatch = context.dispatch
        if dispatch.semantic_op not in self.op_names:
            return None
        if len(dispatch.arguments) != 10 or len(dispatch.outputs) != 2:
            raise CodegenError(
                "GatherReduceQKVRoPEWithCache semantic TIR requires 10 "
                f"arguments and 2 outputs, got {len(dispatch.arguments)} and "
                f"{len(dispatch.outputs)}."
            )
        _validate_contract(dispatch.semantic_attrs)
        _validate_partial_types(dispatch.semantic_attrs)
        implementations = context.implementations(self.family, mode="decode")
        if not implementations:
            raise CodegenError(
                f"Implementation model {context.implementation_model.name!r} "
                "has no decode implementation for "
                "GatherReduceQKVRoPEWithCache."
            )
        candidates = tuple(context.candidate(value) for value in implementations)
        return TIRMicroKernelProposal(
            candidates,
            context.choose_default(self.family, candidates),
        )


def _validate_partial_types(attrs) -> None:
    for name in ("materialized_qkv_type", "logical_qkv_type"):
        value = attrs.get(name)
        if isinstance(value, dict):
            value = type_from_data(value)
        if not isinstance(value, TupleType) or len(value.fields) != 3:
            raise CodegenError(
                f"GatherReduceQKVRoPEWithCache requires a three-field {name}."
            )
        if not all(isinstance(field, DistributedType) for field in value.fields):
            raise CodegenError(
                f"GatherReduceQKVRoPEWithCache {name} fields must be distributed."
            )


__all__ = ["GatherReduceQKVRoPEWithCacheMicroKernelProvider"]
