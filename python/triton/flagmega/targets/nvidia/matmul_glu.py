# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Optional SM90 ordered gate/up scale-block cooperation."""

from triton.flagmega.codegen.triton.implementation import TritonImplementation


def matmul_glu_cooperative_implementations():
    return (
        TritonImplementation(
            "tir.matmul_glu.cooperative_wgmma",
            "matmul_glu",
            "cooperative_wgmma",
            {"tile_n": 16, "block_k": 128, "compute_num_warps": 8, "scaled_accumulation": "fma"},
            {"requires_chip_glu_blocks": True},
            requires=("mma_v3", "fp8", "cooperative_grid", "grid_sync"),
            facts={"portable_triton": False, "internal_grid_barriers": 1, "source_memory_scope": "chip"},
        ),
    )
