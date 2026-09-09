# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Matrix implementations with explicit feature requirements, not graph policy."""

from triton.flagmega.codegen.triton.implementation import TritonImplementation


def delta_rule_implementations():
    return (
        TritonImplementation("tir.delta_rule_gates.local", "delta_rule_gates", "local",
                             parameters={"elements_per_program": 128}, contract={"indexing": "local"}),
        TritonImplementation("tir.delta_rule_block_update.blockwise", "delta_rule_block_update", "blockwise",
                             parameters={"value_tile": 64, "compute_num_warps":
                                         8}, contract={"indexing": "local"}, requires=("mma_v3", )),
        TritonImplementation("tir.delta_rule_coefficients.blockwise", "delta_rule_coefficients", "blockwise",
                             contract={"indexing": "local", "inverse_rounding": "fp16_blocks"}, requires=("mma_v3", )),
        TritonImplementation("tir.delta_rule_log_prefix.grouped", "delta_rule_log_prefix", "grouped",
                             contract={"indexing": "local", "scan_rounding": "grouped_fp32"}),
    )
