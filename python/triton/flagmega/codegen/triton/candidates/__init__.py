# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Reviewed Triton candidate-family providers."""

from triton.flagmega.targets.nvidia.mla import NvidiaMlaProjectionSemanticTIRCandidateProvider
from triton.flagmega.targets.nvidia.kda_input_projection import KdaInputProjectionCandidateProvider
from triton.flagmega.targets.nvidia.partitioned_linear import PartitionedLinearCandidateProvider
from triton.flagmega.targets.nvidia.partitioned_block_scaled_linear import PartitionedBlockScaledLinearCandidateProvider

from .add_norm_stats import AddNormStatsCandidateProvider
from .attention_primitives import AttentionPrimitiveSemanticTIRCandidateProvider
from .core import (
    TritonCandidateContext,
    TritonCandidateProposal,
    TritonCandidateProvider,
    TritonCandidateProviderRegistry,
)
from .delta_rule import DeltaRuleCandidateProvider
from .interleaved_projection import InterleavedProjectionCandidateProvider
from .dense_matmul import DenseMatmulCandidateProvider, MatMulNormStatsCandidateProvider
from .dense_matmul_glu import DenseMatmulGluCandidateProvider
from .distributed_boxing import DistributedBoxingCandidateProvider
from .gather_reduce_add_norm_apply import (
    GatherReduceAddNormApplyCandidateProvider,
)
from .gather_reduce_norm_apply import (
    GatherReduceNormApplySemanticTIRCandidateProvider,
)
from .normalization import (
    GatedNormApplyCandidateProvider,
    NormApplyCandidateProvider,
    NormStatsCandidateProvider,
)
from .packed_qkv import PackedQKVSemanticTIRCandidateProvider
from .paged_attention_split import PagedAttentionSplitSemanticTIRCandidateProvider
from .qkv_rope_with_cache import QKVRoPEWithCacheSemanticTIRCandidateProvider
from .reduction import ReductionCandidateProvider
from .routing import (
    KdaCandidateProvider,
    MhcCandidateProvider,
    StagedRoutingCandidateProvider,
)
from .simple import (
    BlockFp8CandidateProvider,
    ElementwiseCandidateProvider,
    EmbeddingCandidateProvider,
    GdnCandidateProvider,
    GreedySampleCandidateProvider,
    MatmulGluCandidateProvider,
    RmsNormCandidateProvider,
)
from .sparse_experts import SparseExpertsCandidateProvider
from .tensor_transform import TensorTransformCandidateProvider


def default_triton_candidate_registry() -> TritonCandidateProviderRegistry:
    registry = TritonCandidateProviderRegistry()
    for provider in (
            ElementwiseCandidateProvider(),
            TensorTransformCandidateProvider(),
            ReductionCandidateProvider(),
            KdaCandidateProvider(),
            KdaInputProjectionCandidateProvider(),
            PartitionedLinearCandidateProvider(),
            PartitionedBlockScaledLinearCandidateProvider(),
            MhcCandidateProvider(),
            StagedRoutingCandidateProvider(),
            DeltaRuleCandidateProvider(),
            SparseExpertsCandidateProvider(),
            BlockFp8CandidateProvider(),
            InterleavedProjectionCandidateProvider(),
            MatmulGluCandidateProvider(),
            DenseMatmulGluCandidateProvider(),
            EmbeddingCandidateProvider(),
            GreedySampleCandidateProvider(),
            RmsNormCandidateProvider(),
            NormStatsCandidateProvider(),
            NormApplyCandidateProvider(),
            GatedNormApplyCandidateProvider(),
            PackedQKVSemanticTIRCandidateProvider(),
            PagedAttentionSplitSemanticTIRCandidateProvider(),
            AttentionPrimitiveSemanticTIRCandidateProvider(),
            NvidiaMlaProjectionSemanticTIRCandidateProvider(),
            QKVRoPEWithCacheSemanticTIRCandidateProvider(),
            GatherReduceNormApplySemanticTIRCandidateProvider(),
            GatherReduceAddNormApplyCandidateProvider(),
            DenseMatmulCandidateProvider(),
            MatMulNormStatsCandidateProvider(),
            AddNormStatsCandidateProvider(),
            GdnCandidateProvider(),
            DistributedBoxingCandidateProvider(),
    ):
        registry.add(provider)
    return registry


__all__ = [
    "DeltaRuleCandidateProvider",
    "SparseExpertsCandidateProvider",
    "ReductionCandidateProvider",
    "TensorTransformCandidateProvider",
    "BlockFp8CandidateProvider",
    "InterleavedProjectionCandidateProvider",
    "AttentionPrimitiveSemanticTIRCandidateProvider",
    "DenseMatmulCandidateProvider",
    "MatMulNormStatsCandidateProvider",
    "AddNormStatsCandidateProvider",
    "GatherReduceAddNormApplyCandidateProvider",
    "DenseMatmulGluCandidateProvider",
    "ElementwiseCandidateProvider",
    "DistributedBoxingCandidateProvider",
    "EmbeddingCandidateProvider",
    "GdnCandidateProvider",
    "GreedySampleCandidateProvider",
    "MatmulGluCandidateProvider",
    "MhcCandidateProvider",
    "GatedNormApplyCandidateProvider",
    "NormApplyCandidateProvider",
    "NormStatsCandidateProvider",
    "PackedQKVSemanticTIRCandidateProvider",
    "PagedAttentionSplitSemanticTIRCandidateProvider",
    "QKVRoPEWithCacheSemanticTIRCandidateProvider",
    "GatherReduceNormApplySemanticTIRCandidateProvider",
    "RmsNormCandidateProvider",
    "TritonCandidateContext",
    "TritonCandidateProposal",
    "TritonCandidateProvider",
    "TritonCandidateProviderRegistry",
    "default_triton_candidate_registry",
]
