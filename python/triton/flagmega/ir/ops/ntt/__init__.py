# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Target-independent NTT representation operations."""

from triton.flagmega.ir.ops.ntt.add_norm_stats import (
    AddNormStats,
    can_materialize_sum_partial,
)
from triton.flagmega.ir.ops.ntt.gather_reduce_norm_apply import (
    GatherReduceNormApply,
)
from triton.flagmega.ir.ops.ntt.gather_reduce_add_norm_apply import (
    GatherReduceAddNormApply,
)
from triton.flagmega.ir.ops.ntt.matmul_norm_stats import MatMulNormStats
from triton.flagmega.ir.ops.ntt.packed_matmul import PackedMatMul
from triton.flagmega.ir.ops.ntt.packed_qkv_parallel_linear import PackedQKVParallelLinear
from triton.flagmega.ir.ops.ntt.packed_qkv_parallel_linear_combine import (
    PackedQKVParallelLinearCombine,
    can_materialize_packed_qkv,
)
from triton.flagmega.ir.ops.ntt.paged_attention_combine import PagedAttentionCombine
from triton.flagmega.ir.ops.ntt.paged_attention_gated_combine import PagedAttentionGatedCombine
from triton.flagmega.ir.ops.ntt.paged_attention_partial import PagedAttentionPartial
from triton.flagmega.ir.ops.ntt.vectorized_cast import VectorizedCast
from triton.flagmega.ir.ops.ntt.vectorized_rope import VectorizedRoPE
from triton.flagmega.ir.ops.ntt.sparse_experts import DispatchedExpertsGateUp, SparseExpertsDownCombine

__all__ = [
    "DispatchedExpertsGateUp",
    "SparseExpertsDownCombine",
    "GatherReduceAddNormApply",
    "GatherReduceNormApply",
    "AddNormStats",
    "MatMulNormStats",
    "PackedMatMul",
    "PackedQKVParallelLinear",
    "PackedQKVParallelLinearCombine",
    "PagedAttentionCombine",
    "PagedAttentionGatedCombine",
    "PagedAttentionPartial",
    "VectorizedCast",
    "VectorizedRoPE",
    "can_materialize_sum_partial",
    "can_materialize_packed_qkv",
]
