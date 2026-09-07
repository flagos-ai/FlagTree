# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Reviewed Triton candidate-family providers."""

from .core import (
    TritonCandidateContext,
    TritonCandidateProposal,
    TritonCandidateProvider,
    TritonCandidateProviderRegistry,
)
from .dense_matmul import DenseMatmulCandidateProvider, MatMulNormStatsCandidateProvider
from .matmul_norm_stats_combine import MatMulNormStatsCombineCandidateProvider
from .gather_reduce_add_norm_apply import (
    GatherReduceAddNormApplyCandidateProvider,
)
from .dense_matmul_glu import DenseMatmulGluCandidateProvider
from .distributed_boxing import DistributedBoxingCandidateProvider
from .packed_qkv import PackedQKVSemanticTIRCandidateProvider
from .paged_attention_split import PagedAttentionSplitSemanticTIRCandidateProvider
from .qkv_rope_with_cache import QKVRoPEWithCacheSemanticTIRCandidateProvider
from .gather_reduce_qkv_rope_with_cache import (
    GatherReduceQKVRoPEWithCacheSemanticTIRCandidateProvider,
)
from .gather_reduce_norm_apply import (
    GatherReduceNormApplySemanticTIRCandidateProvider,
)
from .attention_primitives import AttentionPrimitiveSemanticTIRCandidateProvider
from .normalization import NormApplyCandidateProvider, NormStatsCandidateProvider
from .simple import (
    BlockFp8CandidateProvider,
    ElementwiseCandidateProvider,
    EmbeddingCandidateProvider,
    GdnCandidateProvider,
    GreedySampleCandidateProvider,
    MatmulGluCandidateProvider,
    RmsNormCandidateProvider,
)


def default_triton_candidate_registry() -> TritonCandidateProviderRegistry:
    registry = TritonCandidateProviderRegistry()
    for provider in (
        ElementwiseCandidateProvider(),
        BlockFp8CandidateProvider(),
        MatmulGluCandidateProvider(),
        DenseMatmulGluCandidateProvider(),
        EmbeddingCandidateProvider(),
        GreedySampleCandidateProvider(),
        RmsNormCandidateProvider(),
        NormStatsCandidateProvider(),
        NormApplyCandidateProvider(),
        PackedQKVSemanticTIRCandidateProvider(),
        PagedAttentionSplitSemanticTIRCandidateProvider(),
        AttentionPrimitiveSemanticTIRCandidateProvider(),
        QKVRoPEWithCacheSemanticTIRCandidateProvider(),
        GatherReduceQKVRoPEWithCacheSemanticTIRCandidateProvider(),
        GatherReduceNormApplySemanticTIRCandidateProvider(),
        GatherReduceAddNormApplyCandidateProvider(),
        DenseMatmulCandidateProvider(),
        MatMulNormStatsCandidateProvider(),
        MatMulNormStatsCombineCandidateProvider(),
        GdnCandidateProvider(),
        DistributedBoxingCandidateProvider(),
    ):
        registry.add(provider)
    return registry


__all__ = [
    "BlockFp8CandidateProvider",
    "AttentionPrimitiveSemanticTIRCandidateProvider",
    "DenseMatmulCandidateProvider",
    "MatMulNormStatsCandidateProvider",
    "MatMulNormStatsCombineCandidateProvider",
    "GatherReduceAddNormApplyCandidateProvider",
    "DenseMatmulGluCandidateProvider",
    "ElementwiseCandidateProvider",
    "DistributedBoxingCandidateProvider",
    "EmbeddingCandidateProvider",
    "GdnCandidateProvider",
    "GreedySampleCandidateProvider",
    "MatmulGluCandidateProvider",
    "NormApplyCandidateProvider",
    "NormStatsCandidateProvider",
    "PackedQKVSemanticTIRCandidateProvider",
    "PagedAttentionSplitSemanticTIRCandidateProvider",
    "QKVRoPEWithCacheSemanticTIRCandidateProvider",
    "GatherReduceQKVRoPEWithCacheSemanticTIRCandidateProvider",
    "GatherReduceNormApplySemanticTIRCandidateProvider",
    "RmsNormCandidateProvider",
    "TritonCandidateContext",
    "TritonCandidateProposal",
    "TritonCandidateProvider",
    "TritonCandidateProviderRegistry",
    "default_triton_candidate_registry",
]
