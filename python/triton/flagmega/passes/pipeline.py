# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Top-level FlagMega pipeline containing implemented passes only.

The pipeline is deliberately an inventory of executable transformations, not
an aspirational copy of nncase pass names. A pass is added here only when its
body performs the transformation named by the pass. Verification remains an
invariant of :class:`Stage` and :class:`PassManager`; it is not repeated under
placeholder optimization names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class _TargetPassRegistrar(Protocol):
    def __call__(self, registry: "PipelinePassRegistry") -> None: ...


@dataclass(frozen=True)
class PipelinePass:
    name: str
    stage: str


@dataclass(frozen=True)
class TargetPipelineExtension:
    """A target-owned pass-registration point inside a generic pipeline."""

    name: str
    registrar: str


class PipelinePassRegistry:
    """Ordered registry populated by target backends, like nncase IRulesAddable."""

    def __init__(self) -> None:
        self._passes: list[PipelinePass] = []

    def add(self, name: str, stage: str) -> None:
        if not name or not stage:
            raise ValueError("A registered pipeline pass requires a name and stage.")
        if any(item.name == name for item in self._passes):
            raise ValueError(f"Pipeline pass {name!r} is already registered.")
        self._passes.append(PipelinePass(name, stage))

    @property
    def passes(self) -> tuple[PipelinePass, ...]:
        return tuple(self._passes)


@dataclass(frozen=True)
class PipelineGroup:
    name: str
    active_stages: frozenset[str]
    output_stage: str
    passes: tuple[PipelinePass | TargetPipelineExtension, ...]


def expand_pipeline_passes(group: PipelineGroup, target) -> tuple[PipelinePass, ...]:
    """Resolve target extension points without exposing machine names to passes."""

    expanded: list[PipelinePass] = []
    for item in group.passes:
        if isinstance(item, PipelinePass):
            expanded.append(item)
            continue
        registrar: _TargetPassRegistrar = getattr(target, item.registrar)
        registry = PipelinePassRegistry()
        registrar(registry)
        expanded.extend(registry.passes)
    return tuple(expanded)


PIPELINE_GROUPS = (
    PipelineGroup(
        "TargetIndependentPass",
        # Compatibility entry points for dumps emitted by the removed
        # placeholder canonical/egraph stages.
        frozenset({
            "imported",
            "canonical",
            "egraph_candidates",
            "extracted",
            "normalization_decomposed",
            "decomposed",
        }),
        "call_invariants_hoisted",
        (
            PipelinePass("DecomposeComplexOps", "decompose-gdn"),
            PipelinePass(
                "FormQKVRoPEWithCache",
                "form-qkv-rope-with-cache",
            ),
            PipelinePass("HoistCallInvariantExpressions", "hoist-call-invariants"),
        ),
    ),
    PipelineGroup(
        "AutoVectorizePass",
        frozenset({"decomposed", "call_invariants_hoisted", "vectorization_candidates"}),
        "vectorized",
        (
            PipelinePass("AutoVectorize", "propose-vectorization"),
            PipelinePass("ApplyVectorization", "apply-vectorization"),
        ),
    ),
    PipelineGroup(
        "AutoPackingPass",
        # ``packed`` is a valid edit-and-resume checkpoint: the packing
        # decisions have already been materialized, but the function ABI
        # threading pass still has to run before AutoDistribution.
        frozenset({
            "vectorized",
            "packing_candidates",
            "packed",
            "boundary_layout_propagated",
            "boundary_layout_cleaned",
            "attention_decomposed",
        }),
        "unused_functions_removed",
        (
            PipelinePass("AutoPacking", "propose-packing"),
            PipelinePass("ApplyPacking", "apply-packing"),
            PipelinePass(
                "FunctionBoundaryLayoutPropagation",
                "propagate-function-boundary-layouts",
            ),
            PipelinePass(
                "PostFunctionBoundaryPackPropagation",
                "post-function-boundary-pack-propagation",
            ),
            PipelinePass(
                "ThreadNormStatsAcrossFunctionBoundaries",
                "thread-norm-stats",
            ),
            TargetPipelineExtension(
                "PostAutoPackingPasses",
                "register_post_auto_packing_passes",
            ),
            PipelinePass(
                "FormMatMulNormStatsCombine",
                "form-matmul-norm-stats-combine",
            ),
            PipelinePass("RemoveUnusedFunctions", "remove-unused-functions"),
        ),
    ),
    PipelineGroup(
        "AutoDistributedPass",
        frozenset({
            "stats_threaded",
            "stats_combined",
            "unused_functions_removed",
            "distributed",
            "distribution_candidates",
            "qkv_combine_folded",
            "qkv_combine_lowered",
            "norm_stats_boxing_sunk",
            "distributed_boundary_layout_propagated",
            "norm_bindings_finalized",
            "finalized_norm_stats_boxing_sunk",
        }),
        "vector_contracts_lowered",
        (
            PipelinePass(
                "FormMatMulNormStatsCombine",
                "form-matmul-norm-stats-combine",
            ),
            PipelinePass("RemoveUnusedFunctions", "remove-unused-functions"),
            PipelinePass(
                "ProposeAutoDistributed",
                "propose-distribution",
            ),
            PipelinePass("AutoDistributed", "auto-distributed"),
            PipelinePass(
                "FoldMaterializedPackedQKVParallelLinearCombine",
                "fold-materialized-packed-qkv-combine",
            ),
            PipelinePass(
                "LowerPackedQKVParallelLinearCombine",
                "lower-packed-qkv-combine",
            ),
            PipelinePass(
                "SinkNormStatsBoxingAcrossFunctionBoundaries",
                "sink-norm-stats-boxing",
            ),
            PipelinePass(
                "PropagatePostAutoDistributedFunctionBoundaryLayouts",
                "propagate-post-auto-distributed-function-boundary-layouts",
            ),
            PipelinePass(
                "FinalizeNormStatsBindings",
                "finalize-norm-stats-bindings",
            ),
            PipelinePass(
                # Boundary output propagation can expose new Partial->B
                # arguments only after the first sinking pass. Folded binding
                # identities then expose their actual NormApply consumers.
                "SinkFinalizedNormStatsBoxingAcrossFunctionBoundaries",
                "sink-finalized-norm-stats-boxing",
            ),
            PipelinePass(
                "LowerMatMulNormStatsCombine",
                "lower-matmul-norm-stats-combine",
            ),
            PipelinePass(
                "LowerVectorizationContracts",
                "lower-vectorization-contracts",
            ),
        ),
    ),
    PipelineGroup(
        "TIRPass",
        frozenset({
            "distribution_candidates",
            "distributed",
            "qkv_combine_folded",
            "qkv_combine_lowered",
            "norm_stats_boxing_sunk",
            "norm_bindings_finalized",
            "vector_contracts_lowered",
            "matmul_norm_stats_lowered",
            "fused_norm",
            "tir_candidates",
            "selected_tir_variants",
            "canonical_constants",
            "frozen_constants",
            "gather_reduce_add_norm_apply_fused",
            "gather_reduce_norm_apply_fused",
            "gather_reduce_qkv_fused",
            "tuple_boxing_lowered",
            "selected_tir",
            "canonicalized_tir",
            "microkernel_candidates",
            "selected_microkernels",
            "packaged_tir",
            "memory_placed_tir",
            "allocated_tir",
            "scheduled_tir",
            "synchronized_tir",
        }),
        "bufferized_tir",
        (
            PipelinePass("FuseNormStatsApply", "fuse-norm-stats-apply"),
            PipelinePass("ConstantCSE", "constant-cse"),
            PipelinePass("FreezeConstantIslands", "freeze-constants"),
            PipelinePass(
                "FuseGatherReduceAddNormApply",
                "fuse-gather-reduce-add-norm-apply",
            ),
            PipelinePass(
                "FuseGatherReduceNormApply",
                "fuse-gather-reduce-norm-apply",
            ),
            PipelinePass(
                "FuseGatherReduceQKVRoPEWithCache",
                "fuse-gather-reduce-qkv-rope-with-cache",
            ),
            PipelinePass("LowerTupleBoxing", "lower-tuple-boxing"),
            PipelinePass("ProposeTIRCandidates", "propose-tir"),
            PipelinePass("LowerSelectedTIR", "lower-tir"),
            PipelinePass(
                "CanonicalizePackedQKVWeights",
                "canonicalize-packed-qkv-weights",
            ),
            PipelinePass("ProposeTIRMicroKernels", "propose-microkernels"),
            PipelinePass("SelectTIRMicroKernels", "select-microkernels"),
            PipelinePass("FinalizeTIRPackage", "finalize-tir-package"),
            PipelinePass("PlanFunctionMemory", "plan-function-memory"),
            PipelinePass("Bufferize", "bufferize"),
            PipelinePass(
                "MaterializeExecutionFunctions",
                "materialize-execution-functions",
            ),
            PipelinePass("PlanMemorySynchronization", "plan-memory-synchronization"),
            PipelinePass(
                "LowerTransferPipelineRegions",
                "lower-transfer-pipeline-regions",
            ),
        ),
    ),
)


__all__ = [
    "PIPELINE_GROUPS",
    "PipelineGroup",
    "PipelinePass",
    "PipelinePassRegistry",
    "TargetPipelineExtension",
    "expand_pipeline_passes",
]
