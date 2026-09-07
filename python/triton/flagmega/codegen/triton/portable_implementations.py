# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Reusable Triton implementations that do not require a vendor feature.

This module is an implementation-library boundary, not a graph policy.  The
entries below may be composed into any target implementation model whose
Triton backend supports the ordinary load/store/reduction operations used by
the corresponding generic templates.  A target remains free to omit them or
prefer a platform-specialized implementation with the same semantic family.
"""

from __future__ import annotations

from collections.abc import Mapping

from triton.flagmega.codegen.triton.implementation import TritonImplementation


def portable_attention_implementations() -> tuple[TritonImplementation, ...]:
    """Return decode implementations for the decomposed attention primitives."""

    portable = {"portable_triton": True}
    return (
        TritonImplementation(
            "tir.rotary_embedding.decode",
            "rotary_embedding",
            "decode",
            {"elements_per_program": 128},
            {"mode": "decode"},
            facts=portable,
        ),
        TritonImplementation(
            "tir.rope.decode",
            "rope",
            "decode",
            {"elements_per_program": 128},
            {"mode": "decode"},
            facts=portable,
        ),
        TritonImplementation(
            "tir.update_paged_attention_kv_cache.decode",
            "update_paged_attention_kv_cache",
            "decode",
            {"elements_per_program": 128},
            {"mode": "decode"},
            facts=portable,
        ),
        TritonImplementation(
            "tir.paged_attention_partial.decode_t16",
            "paged_attention_partial",
            "decode_t16",
            {"token_tile": 16},
            {"mode": "decode"},
            facts=portable,
        ),
        TritonImplementation(
            "tir.paged_attention_partial.decode_t32",
            "paged_attention_partial",
            "decode_t32",
            {"token_tile": 32},
            {"mode": "decode"},
            facts=portable,
        ),
        TritonImplementation(
            "tir.paged_attention_combine.decode",
            "paged_attention_combine",
            "decode",
            {"elements_per_program": 128},
            {"mode": "decode"},
            requires=("cooperative_grid", "grid_sync"),
            facts=portable,
        ),
        TritonImplementation(
            "tir.qkv_rope_with_cache.decode",
            "qkv_rope_with_cache",
            "decode",
            {"elements_per_program": 128},
            {"mode": "decode"},
            facts=portable,
        ),
        TritonImplementation(
            "tir.gather_reduce_qkv_rope_with_cache.decode",
            "gather_reduce_qkv_rope_with_cache",
            "decode",
            {"elements_per_program": 128},
            {"mode": "decode"},
            requires=("cooperative_grid", "grid_sync"),
            facts=portable,
        ),
    )


def portable_attention_preferences() -> Mapping[str, tuple[str, ...]]:
    """Return deterministic fallbacks for the portable attention families."""

    return {
        "rotary_embedding": ("tir.rotary_embedding.decode",),
        "rope": ("tir.rope.decode",),
        "update_paged_attention_kv_cache": (
            "tir.update_paged_attention_kv_cache.decode",
        ),
        "paged_attention_partial": (
            "tir.paged_attention_partial.decode_t16",
            "tir.paged_attention_partial.decode_t32",
        ),
        "paged_attention_combine": ("tir.paged_attention_combine.decode",),
        "qkv_rope_with_cache": ("tir.qkv_rope_with_cache.decode",),
        "gather_reduce_qkv_rope_with_cache": (
            "tir.gather_reduce_qkv_rope_with_cache.decode",
        ),
    }


def portable_local_reduction_implementations() -> tuple[TritonImplementation, ...]:
    """Return owner-local value/statistics implementations.

    These kernels consume only the current owner's local shard.  Distributed
    partial materialization, when required, remains an explicit Boxing call;
    the implementation therefore has no target or mesh-size dependency.
    """

    return (
        TritonImplementation(
            "tir.add_norm_stats.local_partial_rms",
            "add_norm_stats",
            "local_partial_rms",
            {"tile": 128},
            {
                "input_kind": "materialized",
                "axis_kind": "last",
                "use_mean": False,
                "output_layout": "distributed",
            },
            facts={"portable_triton": True, "owner_local_stats": True},
        ),
    )


def portable_local_reduction_preferences() -> Mapping[str, tuple[str, ...]]:
    return {
        "add_norm_stats": ("tir.add_norm_stats.local_partial_rms",),
    }


def portable_normalization_implementations() -> tuple[TritonImplementation, ...]:
    """Return cross-owner statistics consumers using ordinary Triton loads."""

    return (
        TritonImplementation(
            "tir.gather_reduce_add_norm_stats.local_shard_sum_rms",
            "gather_reduce_add_norm_stats",
            "sum_rms",
            {"tile": 64, "partial_reduction_width": 16},
            {
                "reduction": "sum",
                "axis_kind": "last",
                "use_mean": False,
                "output_layout": "distributed",
            },
            requires=("cooperative_grid", "grid_sync"),
            facts={
                "portable_triton": True,
                "cross_owner_read": True,
                "distributed_output": True,
                "internal_grid_barriers": 1,
            },
        ),
        TritonImplementation(
            "tir.gather_reduce_add_norm_apply.sum",
            "gather_reduce_add_norm_apply",
            "sum",
            {
                "tile": 16,
                "partial_reduction_width": 32,
                "reduction_width": 128,
            },
            {
                "reduction": "sum",
                "axis_kind": "suffix",
                "supports_mean": True,
                "outer_rows": "single",
            },
            requires=("cooperative_grid",),
            facts={
                "portable_triton": True,
                "cross_owner_read": True,
                "internal_grid_barriers": 1,
            },
        ),
        TritonImplementation(
            "tir.gather_reduce_norm_apply.sum",
            "gather_reduce_norm_apply",
            "sum",
            {"block_size": 1024, "reduction_width": 32},
            {"supports_mean": True, "axis_kind": "suffix"},
            requires=("cooperative_grid",),
            facts={
                "portable_triton": True,
                "cross_owner_read": True,
                "internal_grid_barriers": 0,
            },
        ),
    )


def portable_normalization_preferences() -> Mapping[str, tuple[str, ...]]:
    return {
        "gather_reduce_add_norm_stats": (
            "tir.gather_reduce_add_norm_stats.local_shard_sum_rms",
        ),
        "gather_reduce_add_norm_apply": (
            "tir.gather_reduce_add_norm_apply.sum",
        ),
        "gather_reduce_norm_apply": (
            "tir.gather_reduce_norm_apply.sum",
        ),
    }


__all__ = [
    "portable_attention_implementations",
    "portable_attention_preferences",
    "portable_local_reduction_implementations",
    "portable_local_reduction_preferences",
    "portable_normalization_implementations",
    "portable_normalization_preferences",
]
