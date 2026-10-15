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


def portable_sparse_experts_implementations() -> tuple[TritonImplementation, ...]:
    """SIMT selected-expert projections without moving route rounding across K."""
    return tuple(
        TritonImplementation(
            f"tir.{family}.simt", family, "simt",
            {"block_n": 8, "block_k": 128},
            {"indexing": "local", "rounding": "explicit"},
            facts={"portable_triton": True},
        ) for family in ("sparse_experts_gate_up", "sparse_experts_down", "sparse_experts_dispatch", "sparse_experts_weighted_sum"))


def portable_tensor_transform_implementations() -> tuple[TritonImplementation, ...]:
    ordinary = tuple(
        TritonImplementation(
            f"tir.{family}.local",
            family,
            "local",
            {"elements_per_program": 128},
            {"indexing": "local"},
            facts={"portable_triton": True},
        ) for family in ("pad", "slice", "pack", "unpack", "broadcast_to"))
    concat = tuple(
        TritonImplementation(
            f"tir.concat.local{tile}",
            "concat",
            "local",
            {"elements_per_program": tile},
            {"indexing": "local"},
            facts={"portable_triton": True},
        ) for tile in (128, 2048))
    return ordinary + concat


def portable_axis_reduction_implementations() -> tuple[TritonImplementation, ...]:
    """Tiled axis reductions, without a fixed maximum reduction extent."""
    return tuple(
        TritonImplementation(
            f"tir.{family}.local",
            family,
            "local",
            {"elements_per_program": 256},
            {"indexing": "local"},
            facts={"portable_triton": True},
        ) for family in ("softmax", "reduce_sum", "top_k", "l2_normalization"))



def portable_staged_routing_implementations() -> tuple[TritonImplementation, ...]:
    """Single-tile staged routing; the tile must cover the whole router extent."""
    return tuple(
        TritonImplementation(
            f"tir.staged_routing.local{tile}",
            "staged_routing",
            "local",
            {"elements_per_program": tile},
            {"indexing": "local"},
            facts={"portable_triton": True},
        ) for tile in (128, 256, 512, 1024))


def portable_kda_implementations() -> tuple[TritonImplementation, ...]:
    """GLM KDA decode operators over compact scalar decode buffers."""
    return (
        TritonImplementation("tir.kda_convolution.local", "kda_convolution", "local",
                             {"elements_per_program": 1024}, {"indexing": "local"},
                             facts={"portable_triton": True}),
        TritonImplementation("tir.kda_recurrent_core.local", "kda_recurrent_core", "local",
                             {"elements_per_program": 128}, {"indexing": "local"},
                             facts={"portable_triton": True}),
        TritonImplementation("tir.kda_gated_recurrent_core.local", "kda_gated_recurrent_core",
                             "local", {"elements_per_program": 128}, {"indexing": "local"},
                             facts={"portable_triton": True}),
    )


def portable_mhc_implementations() -> tuple[TritonImplementation, ...]:
    """GLM-5 hyper-connection stages over replicated decode buffers.

    ``elements_per_program`` bounds the hidden/flattened tile; both renderers
    shrink it further so the staged weight or product block fits in registers.
    """
    return (
        TritonImplementation("tir.mhc_pre.local", "mhc_pre", "local",
                             {"elements_per_program": 256}, {"indexing": "local"},
                             facts={"portable_triton": True}),
        TritonImplementation("tir.mhc_post.local", "mhc_post", "local",
                             {"elements_per_program": 256}, {"indexing": "local"},
                             facts={"portable_triton": True}),
    )


def portable_gather_implementations() -> tuple[TritonImplementation, ...]:
    """Owner-local runtime-index gathers over scalar (unpacked) elements."""
    return (TritonImplementation(
        "tir.gather.local",
        "gather",
        "local",
        {"elements_per_program": 128},
        {"indexing": "local"},
        facts={"portable_triton": True},
    ),)


def portable_router_elementwise_implementations() -> tuple[TritonImplementation, ...]:
    """Owner-local division/sigmoid, including typed-vector sigmoid schedules."""
    return tuple(
        TritonImplementation(
            f"tir.elementwise.{variant}.{kind}",
            "elementwise",
            variant,
            {"elements_per_program": 128} if kind == "scalar" else {"vector_groups": 32},
            {"semantic_op": f"math.{variant}", "vectorization_kind": kind},
            facts={"portable_triton": True},
        ) for variant, kind in (("div", "scalar"), ("exp", "scalar"), ("exp", "axes"), ("sigmoid", "scalar"), ("sigmoid", "axes")))


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
            requires=("cooperative_grid", "grid_sync"),
            facts={**portable, "internal_grid_barriers": 1},
        ),
        TritonImplementation(
            "tir.update_paged_attention_kv_cache_pair.decode",
            "update_paged_attention_kv_cache_pair",
            "decode",
            {"elements_per_program": 128},
            {"mode": "decode"},
            requires=("cooperative_grid", "grid_sync"),
            facts={**portable, "internal_grid_barriers": 1},
        ),
        TritonImplementation(
            "tir.dsa_indexer.decode",
            "dsa_indexer",
            "decode",
            {"elements_per_program": 128},
            {"mode": "decode"},
            requires=("cooperative_grid", "grid_sync"),
            facts={**portable, "internal_grid_barriers": 2},
        ),
        TritonImplementation(
            "tir.dsa_paged_attention.decode",
            "dsa_paged_attention",
            "decode",
            {"token_tile": 2048},
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
        # Warp-layout variants of the same token tiles: these choose how a
        # token row's 128-wide reduction maps onto warps rather than how many
        # tokens a program covers, so they are independent of `token_tile`.
        TritonImplementation(
            "tir.paged_attention_partial.decode_layout4",
            "paged_attention_partial",
            "decode_layout4",
            {"token_tile": 16},
            {"mode": "decode"},
            facts=portable,
        ),
        TritonImplementation(
            "tir.paged_attention_partial.decode_layout32",
            "paged_attention_partial",
            "decode_layout32",
            {"token_tile": 32},
            {"mode": "decode"},
            facts=portable,
        ),
        # Larger token tiles. These templates
        # (kernels/paged_attention_partial/decode_t128.py.jinja,
        # decode_t256.py.jinja, decode_layout4/32 above and
        # decode_t128_layout32 below) were in the tree but had no registration
        # entry, so they were unselectable. ITERATION.md's t128 measurement
        # records 13.276/13.289/14.374/16.330 ms/token at contexts
        # 1/128/1024/2048 against decode_t16's 12.128/13.119/20.623/29.114 --
        # t16 wins at short context and collapses at long, t128 is flat.
        TritonImplementation(
            "tir.paged_attention_partial.decode_t128",
            "paged_attention_partial",
            "decode_t128",
            {"token_tile": 128},
            {"mode": "decode"},
            facts=portable,
        ),
        TritonImplementation(
            "tir.paged_attention_partial.decode_t256",
            "paged_attention_partial",
            "decode_t256",
            {"token_tile": 256},
            {"mode": "decode"},
            facts=portable,
        ),
        TritonImplementation(
            "tir.paged_attention_partial.decode_t128_layout32",
            "paged_attention_partial",
            "decode_t128_layout32",
            {"token_tile": 128},
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
            "tir.gather_reduce_qkv_rope_with_cache.decode",
            "gather_reduce_qkv_rope_with_cache",
            "decode",
            {"elements_per_program": 128},
            {"mode": "decode"},
            requires=("cooperative_grid", "grid_sync"),
            facts={**portable, "internal_grid_barriers": 1},
        ),
        TritonImplementation(
            "tir.qkv_rope_with_cache.decode",
            "qkv_rope_with_cache",
            "decode",
            {"elements_per_program": 128},
            {"mode": "decode"},
            requires=("cooperative_grid", "grid_sync"),
            facts={**portable, "internal_grid_barriers": 1},
        ),
        TritonImplementation(
            "tir.paged_attention_gated_combine.decode",
            "paged_attention_gated_combine",
            "decode",
            {"elements_per_program": 128},
            {"mode": "decode"},
            requires=("cooperative_grid", "grid_sync"),
            facts=portable,
        ),
        *(TritonImplementation(
            f"tir.paged_attention_gated_combine.decode_t{tile}",
            "paged_attention_gated_combine", "decode",
            {"elements_per_program": tile}, {"mode": "decode"},
            requires=("cooperative_grid", "grid_sync"), facts=portable,
        ) for tile in (32, 64)),
    )


def portable_attention_preferences() -> Mapping[str, tuple[str, ...]]:
    """Return deterministic fallbacks for the portable attention families."""

    return {
        "rotary_embedding": ("tir.rotary_embedding.decode",),
        "rope": ("tir.rope.decode",),
        "update_paged_attention_kv_cache": (
            "tir.update_paged_attention_kv_cache.decode",
        ),
        # decode_t16 stays first: it is the measured best at short context,
        # which is what batch-one decode serves. The rest are ordered so a
        # target that filters out earlier entries still lands on a legal tile.
        "update_paged_attention_kv_cache_pair": ("tir.update_paged_attention_kv_cache_pair.decode",),
        "dsa_indexer": ("tir.dsa_indexer.decode",),
        "dsa_paged_attention": ("tir.dsa_paged_attention.decode",),
        "paged_attention_partial": (
            "tir.paged_attention_partial.decode_t16",
            "tir.paged_attention_partial.decode_layout4",
            "tir.paged_attention_partial.decode_layout32",
            "tir.paged_attention_partial.decode_t32",
            "tir.paged_attention_partial.decode_t256",
            "tir.paged_attention_partial.decode_t128",
            "tir.paged_attention_partial.decode_t128_layout32",
        ),
        "paged_attention_combine": ("tir.paged_attention_combine.decode",),
        "paged_attention_gated_combine": (
            "tir.paged_attention_gated_combine.decode",
            "tir.paged_attention_gated_combine.decode_t64",
            "tir.paged_attention_gated_combine.decode_t32",
        ),
        "qkv_rope_with_cache": ("tir.qkv_rope_with_cache.decode",),
        "gather_reduce_qkv_rope_with_cache": ("tir.gather_reduce_qkv_rope_with_cache.decode",),
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
    "portable_sparse_experts_implementations",
    "portable_axis_reduction_implementations",
    "portable_tensor_transform_implementations",
    "portable_gather_implementations",
    "portable_staged_routing_implementations",
    "portable_kda_implementations",
    "portable_mhc_implementations",
    "portable_router_elementwise_implementations",
    "portable_attention_implementations",
    "portable_attention_preferences",
    "portable_local_reduction_implementations",
    "portable_local_reduction_preferences",
    "portable_normalization_implementations",
    "portable_normalization_preferences",
]
