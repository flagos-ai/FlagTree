# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""NTT-oriented high-level rewrite rules."""

from triton.flagmega.rules.ntt.vectorize import VectorizeRuleRegistry
from triton.flagmega.rules.ntt.lower_add_norm_stats import (
    lower_add_norm_stats_rule,
)
from triton.flagmega.rules.ntt.packed_qkv_parallel_linear_combine import (
    fold_materialized_packed_qkv_parallel_linear_combine_rule,
    lower_packed_qkv_parallel_linear_combine_rule,
)
from triton.flagmega.rules.ntt.decompose_paged_attention import (
    decompose_paged_attention_rule,
)

__all__ = [
    "VectorizeRuleRegistry",
    "decompose_paged_attention_rule",
    "fold_materialized_packed_qkv_parallel_linear_combine_rule",
    "lower_add_norm_stats_rule",
    "lower_packed_qkv_parallel_linear_combine_rule",
]
