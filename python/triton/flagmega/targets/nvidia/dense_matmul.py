# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Optional ordered BF16 MMA schedules with CTA-private Shared staging."""

from triton.flagmega.codegen.triton.implementation import TritonImplementation


def packed_mma_accumulator_implementations():
    """Expose the FP32 accumulator without an intermediate BF16 store.

    ``accumulation_splits`` partitions complete K tiles, then merges the FP32
    partials in ascending order. A profile may request this algorithm explicitly;
    it is not a cross-CTA or cross-chip reduction.
    """
    return (
        TritonImplementation(
            "tir.dense_matmul.packed_k_major_mma_accumulator",
            "dense_matmul",
            "packed_k_major_mma_accumulator",
            {"block_k": 128, "tile_n": 16, "compute_num_warps": 8, "accumulation_splits": 1,
             "weight_load_layout": "coalesced"},
            {
                "supports_local_row_loop": True,
                "input_kind": "packed",
                "epilogue": "none",
                "vectorization_kind": "output_axis",
                "distribution_kind": "canonical",
                "packed_layout": "k_major_n8_k16",
                "required_input_dtype": "bfloat16",
                "requires_unpartitioned_reduction": True,
                "supports_masked_tiles": True,
            },
            requires=("mma_v3",),
            facts={"portable_triton": False, "ordered_mma_accumulation": True},
        ),
    )


def packed_mma_prefetch_implementations():
    """Keep K32 MMA accumulation while batching global operand fetches."""
    return (
        TritonImplementation(
            "tir.dense_matmul.packed_k_major_mma_prefetch",
            "dense_matmul",
            "packed_k_major_mma_prefetch",
            {"block_k": 32, "tile_n": 16, "compute_num_warps": 8, "prefetch_elements": 16384, "prefetch_max_k": 2048},
            {
                "supports_local_row_loop": True,
                "input_kind": "packed",
                "epilogue": "none",
                "vectorization_kind": "output_axis",
                "distribution_kind": "canonical",
                "packed_layout": "k_major_n8_k16",
                "required_input_dtype": "bfloat16",
                "required_output_dtype": "bfloat16",
                "requires_unpartitioned_reduction": True,
                "supports_masked_tiles": True,
            },
            requires=("mma_v3",),
            facts={"portable_triton": False, "ordered_mma_accumulation": True},
        ),
    )


def packed_mma_norm_stats_implementations():
    """Preserve ordered projection stores inside a multirow RMS epilogue."""
    return (
        TritonImplementation(
            "tir.dense_matmul.packed_k_major_mma_norm_stats", "dense_matmul", "packed_k_major_mma_norm_stats",
            {"block_k": 128, "tile_n": 16, "compute_num_warps": 8,
             "accumulation_splits": 1, "weight_load_layout": "coalesced"},
            {"supports_local_row_loop": True, "input_kind": "packed", "epilogue": "residual_norm_stats",
             "vectorization_kind": "output_axis", "distribution_kind": "canonical",
             "packed_layout": "k_major_n8_k16", "required_input_dtype": "bfloat16",
             "requires_unpartitioned_reduction": True, "supports_masked_tiles": True,
             "statistics_kind": "local_shard"},
            requires=("mma_v3",),
            facts={"ordered_mma_accumulation": True, "owner_local_stats": True, "internal_grid_barriers": 0},
        ),
    )


def ordered_mma_glu_implementations():
    """Fuse two ordered projections while retaining their BF16 stores."""
    return tuple(TritonImplementation(
        "tir.dense_matmul_glu.ordered_mma_" + kind, "dense_matmul_glu", "ordered_mma",
        {"block_k": 128, "tile_n": 16, "compute_num_warps": 8, "accumulation_splits": 1},
        {"input_kind": kind, "fusion": "none", **({"packed_layout": "k_major_n8_k16"} if kind == "packed" else {})},
        requires=("mma_v3",), facts={"ordered_mma_accumulation": True},
    ) for kind in ("logical", "packed"))
