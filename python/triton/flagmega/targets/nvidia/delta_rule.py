# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Hopper block updates reuse resident BF16 state and Q/K descriptors."""

from triton.flagmega.codegen.triton.implementation import TritonImplementation


def resident_delta_rule_implementations():
    return tuple(
        TritonImplementation(
            f"tir.{family}.resident_wgmma", family, "resident_wgmma",
            parameters={"value_tile": 64, "compute_num_warps": 8},
            contract={"indexing": "local"}, requires=("mma_v3",),
            facts={"resident_state_operand": True},
        )
        for family in ("delta_rule_block_update", "delta_rule_chunked_block_update")
    )
