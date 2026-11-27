# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Selected BF16 experts retain an FP32 MMA fragment across all K tiles."""

from triton.flagmega.codegen.triton.implementation import TritonImplementation


def ordered_sparse_experts_implementations():
    return tuple(
        TritonImplementation(
            f"tir.{family}.ordered_mma", family, "ordered_mma",
            parameters={"block_n": 16, "block_k": 128, "compute_num_warps": 8},
            contract={"indexing": "local", "rounding": "explicit", "requires_bf16": True},
            requires=("mma_v3",), facts={"ordered_mma_accumulation": True},
        )
        for family in ("sparse_experts_gate_up", "sparse_experts_down")
    )
