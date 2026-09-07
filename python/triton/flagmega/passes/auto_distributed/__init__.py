# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""AutoDistributed public pass namespace."""

from triton.flagmega.passes.auto_distributed.auto_distributed import AutoDistributedPass
from triton.flagmega.passes.auto_distributed.candidates import (
    DistributedCandidate,
    DistributedCandidateContext,
    DistributedCandidateProvider,
    DistributedCandidateProviderRegistry,
    DistributedCandidateTuple,
    DistributedCandidateProviderBase,
)
from triton.flagmega.passes.auto_distributed.materializer import DistributedMaterializer
from triton.flagmega.passes.auto_distributed.packed_matmul_provider import PackedMatMulCandidateProvider
from triton.flagmega.passes.auto_distributed.inference_providers import (
    TypeInferenceCandidateProvider,
)
from triton.flagmega.passes.auto_distributed.norm_providers import (
    BindNormStatsCandidateProvider,
    MatMulNormStatsCombineCandidateProvider,
    NormApplyCandidateProvider,
    NormStatsCandidateProvider,
)
from triton.flagmega.passes.auto_distributed.operation_cost import (
    DistributedOperationCostModel,
)
from triton.flagmega.passes.auto_distributed.paged_attention_providers import (
    PagedAttentionCombineCandidateProvider,
    PagedAttentionPartialCandidateProvider,
)
from triton.flagmega.passes.auto_distributed.reshard import (
    DistributedReshardPlan,
    DistributedReshardPlanner,
    can_box,
    reshard_plan_cost,
    reshard_step_cost,
)
from triton.flagmega.passes.auto_distributed.reshard_decomposition import (
    get_partial_reduce_scatter_intermediates,
)
from triton.flagmega.passes.auto_distributed.reshard_cost import (
    DistributedReshardCostModel,
)
from triton.flagmega.passes.auto_distributed.realization import (
    CanonicalStorageReshardRealizationPolicy,
    DistributedReshardRealization,
    DistributedReshardRealizationContext,
    DistributedReshardRealizationPolicy,
    DistributedReshardSourceKind,
    DistributedReshardUsageKind,
    NttDistributedReshardRealizationPolicy,
    PyNttDistributedReshardRealizationPolicy,
)
from triton.flagmega.passes.auto_distributed.qkv_rope_with_cache_provider import (
    QKVRoPEWithCacheCandidateProvider,
)
from triton.flagmega.passes.auto_distributed.search import (
    CandidateBucket,
    ReshardSite,
    SearchGraph,
    SearchResult,
    build_search_graph,
    function_boundary_id,
    graph_dot,
    solve_search_graph,
)
from triton.flagmega.passes.auto_distributed.policy import (
    NttDistributionPolicy,
    lower_vectorization_contracts,
)

__all__ = [
    "PackedMatMulCandidateProvider",
    "AutoDistributedPass",
    "BindNormStatsCandidateProvider",
    "MatMulNormStatsCombineCandidateProvider",
    "CandidateBucket",
    "CanonicalStorageReshardRealizationPolicy",
    "ReshardSite",
    "DistributedCandidate",
    "DistributedCandidateContext",
    "DistributedCandidateProvider",
    "DistributedCandidateProviderRegistry",
    "DistributedCandidateTuple",
    "DistributedCandidateProviderBase",
    "DistributedMaterializer",
    "DistributedOperationCostModel",
    "TypeInferenceCandidateProvider",
    "NttDistributionPolicy",
    "NormApplyCandidateProvider",
    "NormStatsCandidateProvider",
    "PagedAttentionCombineCandidateProvider",
    "PagedAttentionPartialCandidateProvider",
    "DistributedReshardPlan",
    "DistributedReshardPlanner",
    "DistributedReshardCostModel",
    "DistributedReshardRealization",
    "DistributedReshardRealizationContext",
    "DistributedReshardRealizationPolicy",
    "DistributedReshardSourceKind",
    "DistributedReshardUsageKind",
    "NttDistributedReshardRealizationPolicy",
    "PyNttDistributedReshardRealizationPolicy",
    "QKVRoPEWithCacheCandidateProvider",
    "SearchGraph",
    "SearchResult",
    "build_search_graph",
    "can_box",
    "get_partial_reduce_scatter_intermediates",
    "function_boundary_id",
    "graph_dot",
    "lower_vectorization_contracts",
    "reshard_plan_cost",
    "reshard_step_cost",
    "solve_search_graph",
]
