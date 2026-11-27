# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""AutoVectorize rules mirroring nncase's NTT rule grouping."""

from triton.flagmega.rules.ntt.vectorize.base import VectorizeCandidate, VectorizeRule, VectorizeRuleRegistry
from triton.flagmega.rules.ntt.vectorize.binary import VectorizeBinary
from triton.flagmega.rules.ntt.vectorize.matmul import VectorizeMatMul
from triton.flagmega.rules.ntt.vectorize.norm import VectorizeRMSNorm
from triton.flagmega.rules.ntt.vectorize.norm_apply import VectorizeNormApply
from triton.flagmega.rules.ntt.vectorize.norm_stats import VectorizeNormStats
from triton.flagmega.rules.ntt.vectorize.policy import NttVectorizationPolicy
from triton.flagmega.rules.ntt.vectorize.propagation import propagation_rules
from triton.flagmega.rules.ntt.vectorize.qkv_rope_with_cache import VectorizeQKVRoPEWithCache
from triton.flagmega.rules.ntt.vectorize.unary import VectorizeUnary

__all__ = [
    "NttVectorizationPolicy", "VectorizeBinary",
    "VectorizeCandidate", "VectorizeMatMul", "VectorizeNormApply", "VectorizeNormStats",
    "VectorizeQKVRoPEWithCache", "VectorizeRule", "VectorizeRMSNorm",
    "VectorizeRuleRegistry", "VectorizeUnary", "propagation_rules",
]
