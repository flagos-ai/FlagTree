# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Composable stage registry and built-in FlagMega pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from triton.flagmega.errors import StageError
from triton.flagmega.ir import IRModule, ProvenanceRecord, verify_module
from triton.flagmega.passes.target_independent import decompose_complex_ops
from triton.flagmega.passes.tir.fuse_distributed_ops import fuse_distributed_ops
from triton.flagmega.passes.norm_stats import (
    finalize_norm_stats_bindings,
    fuse_norm_stats_apply,
    lower_matmul_norm_stats_combine,
)
from triton.flagmega.passes.functions import (
    form_matmul_norm_stats_combine,
    hoist_call_invariant_expressions,
    lift_constant_parameter_expressions,
    post_function_boundary_pack_propagation,
    propagate_function_boundary_layouts,
    propagate_post_auto_distributed_function_boundary_layouts,
    remove_unused_functions,
    sink_norm_stats_boxing_across_function_boundaries,
    thread_norm_stats_across_function_boundaries,
)
from triton.flagmega.passes.constants import ConstantCSEPass, FreezeConstantIslandsPass
from triton.flagmega.passes.packed_qkv_combine import (
    fold_materialized_packed_qkv_parallel_linear_combine,
    lower_packed_qkv_parallel_linear_combine,
)
from triton.flagmega.passes.auto_distributed.policy import (
    lower_vectorization_contracts,
)
from triton.flagmega.passes.tir import (
    canonicalize_packed_qkv_weights,
    lower_transfer_pipeline_regions,
    lower_tuple_boxing,
    materialize_execution_functions,
)
from triton.flagmega.rules import DataflowRewriter
from triton.flagmega.rules.neutral import auto_packing_neutral_rules
from triton.flagmega.rules.ntt.decompose_paged_attention import (
    decompose_paged_attention_rule,
    paged_attention_split_plan,
)
from triton.flagmega.codegen.triton.kernel_dispatch import selected_kernel_nodes
from triton.flagmega.targets.base import Target


StageTransform = Callable[[IRModule, Target], IRModule]


@dataclass(frozen=True)
class Stage:
    name: str
    input_stage: str
    output_stage: str
    transform: StageTransform
    output_dialect: str | None = None
    selection_point: bool = False
    compatible_input_stages: frozenset[str] = frozenset()

    def accepts(self, module_stage: str) -> bool:
        return module_stage == self.input_stage or module_stage in self.compatible_input_stages

    def run(self, module: IRModule, target: Target) -> IRModule:
        verify_module(module)
        if not self.accepts(module.stage):
            accepted = sorted({self.input_stage, *self.compatible_input_stages})
            raise StageError(
                f"Stage {self.name!r} requires one of {accepted}, got {module.stage!r}.",
                stage=module.stage,
            )
        parent_hash = module.semantic_hash
        result = self.transform(module, target)
        result = replace(
            result,
            stage=self.output_stage,
            dialect=self.output_dialect or result.dialect,
            provenance=result.provenance + (
                ProvenanceRecord(self.output_stage, parent_hash, f"flagmega.stage:{self.name}"), ),
        )
        verify_module(result, expected_stage=self.output_stage)
        target.verify(result)
        return result


_STAGES: dict[str, Stage] = {}
_STAGE_ALIASES = {
    "form-qkv-rope-with-cache": "decompose-gdn",
    "fuse-gather-reduce-add-norm-apply": "fuse-distributed-ops",
    "fuse-gather-reduce-norm-apply": "fuse-distributed-ops",
    "fuse-gather-reduce-qkv-rope-with-cache": "fuse-distributed-ops",
}


def register_stage(stage: Stage) -> None:
    if stage.name in _STAGES:
        raise ValueError(f"Stage {stage.name!r} is already registered.")
    _STAGES[stage.name] = stage


def get_stage(name: str) -> Stage:
    try:
        return _STAGES[_STAGE_ALIASES.get(name, name)]
    except KeyError as error:
        raise StageError(f"Unknown FlagMega stage {name!r}; available: {sorted(_STAGES)}") from error


def next_stage(module_stage: str) -> Stage | None:
    exact = [
        stage for stage in _STAGES.values()
        if stage.input_stage == module_stage
    ]
    matches = exact or [
        stage for stage in _STAGES.values() if stage.accepts(module_stage)
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise StageError(f"Stage {module_stage!r} has ambiguous successors: {[value.name for value in matches]}.")
    return matches[0]


def stage_names() -> tuple[str, ...]:
    return tuple(sorted({*_STAGES, *_STAGE_ALIASES}))


def _propose_distribution(module: IRModule, target: Target) -> IRModule:
    # Direct stage invocation may resume an older ``stats_combined`` Python
    # checkpoint.  The grouped pipeline runs the named removal pass first;
    # keep the stage itself total for edit-and-resume callers as well.
    return target.propose_distribution(remove_unused_functions(module))


def _auto_distribute(module: IRModule, target: Target) -> IRModule:
    if module.stage == "distribution_candidates":
        return target.apply_distribution(module)
    return target.auto_distribute(remove_unused_functions(module))


def _lower_vectorization_contracts(
    module: IRModule, _target: Target
) -> IRModule:
    return lower_vectorization_contracts(module)


def _propose_vectorization(module: IRModule, target: Target) -> IRModule:
    return target.propose_vectorization(module)


def _apply_vectorization(module: IRModule, target: Target) -> IRModule:
    return target.apply_vectorization(module)


def _propose_packing(module: IRModule, target: Target) -> IRModule:
    return target.propose_packing(module)


def _apply_packing(module: IRModule, target: Target) -> IRModule:
    packed = target.apply_packing(module)
    return DataflowRewriter(auto_packing_neutral_rules()).rewrite(packed)


def _decompose_paged_attention(module: IRModule, target: Target) -> IRModule:
    split_axis, split_count = paged_attention_split_plan(
        target.distributed_placements(module)
    )
    return DataflowRewriter((
        decompose_paged_attention_rule(split_axis, split_count),
    )).rewrite(module)


def _propose_tir(module: IRModule, target: Target) -> IRModule:
    # A direct stage call can resume a pre-normalization Python checkpoint.
    # The grouped pipeline exposes the transformation and its own dumps.
    return target.propose_tir(lower_tuple_boxing(module))


def _lower_to_tir(module: IRModule, target: Target) -> IRModule:
    return target.lower_to_tir(module)


def _canonicalize_packed_qkv_weights(
    module: IRModule, _target: Target
) -> IRModule:
    return canonicalize_packed_qkv_weights(module)


def _propose_microkernels(module: IRModule, target: Target) -> IRModule:
    return target.propose_microkernels(module)


def _select_microkernels(module: IRModule, target: Target) -> IRModule:
    return target.select_microkernels(module)


def _finalize_tir_package(module: IRModule, target: Target) -> IRModule:
    kernels = selected_kernel_nodes(module)
    launch_contract = {
        **dict(module.metadata.get("launch_contract", {})),
        **target.plan_launch(module, kernels),
    }
    with_launch = replace(
        module,
        metadata={
            **dict(module.metadata),
            "launch_contract": launch_contract,
        },
    )
    package_plan = target.plan_codegen_package(
        with_launch, kernels
    )
    return replace(
        with_launch,
        metadata={
            **dict(with_launch.metadata),
            "codegen_package_plan": package_plan,
        },
    )


def _constant_cse(module: IRModule, _target: Target) -> IRModule:
    return ConstantCSEPass().run(module)


def _freeze_constants(module: IRModule, _target: Target) -> IRModule:
    return FreezeConstantIslandsPass().run(module)


def _bufferize(module: IRModule, target: Target) -> IRModule:
    return target.bufferize(module)


def _plan_function_memory(module: IRModule, target: Target) -> IRModule:
    return target.plan_function_memory(module)


def _plan_memory_synchronization(module: IRModule, target: Target) -> IRModule:
    return target.plan_memory_synchronization(module)


def _materialize_execution_functions(
    module: IRModule, _target: Target
) -> IRModule:
    return materialize_execution_functions(module)


def _lower_transfer_pipeline_regions(
    module: IRModule, _target: Target
) -> IRModule:
    return lower_transfer_pipeline_regions(module)


register_stage(Stage(
    "decompose-gdn",
    "imported",
    "decomposed",
    lambda module, _target: decompose_complex_ops(module),
    compatible_input_stages=frozenset({"canonical", "egraph_candidates", "extracted", "normalization_decomposed"}),
))
register_stage(Stage(
    "hoist-call-invariants",
    "decomposed",
    "call_invariants_hoisted",
    lambda module, _target: hoist_call_invariant_expressions(module),
))
register_stage(Stage(
    "propose-vectorization",
    "call_invariants_hoisted",
    "vectorization_candidates",
    _propose_vectorization,
    selection_point=True,
    compatible_input_stages=frozenset({"decomposed"}),
))
register_stage(Stage(
    "apply-vectorization",
    "vectorization_candidates",
    "vectorized",
    _apply_vectorization,
))
register_stage(
    Stage(
        "propose-packing",
        "vectorized",
        "packing_candidates",
        _propose_packing,
        selection_point=True,
    ))
register_stage(Stage("apply-packing", "packing_candidates", "packed", _apply_packing))
register_stage(Stage(
    "thread-norm-stats",
    "boundary_layout_cleaned",
    "stats_threaded",
    lambda module, _target: thread_norm_stats_across_function_boundaries(module),
))
register_stage(Stage(
    "propagate-function-boundary-layouts",
    "packed",
    "boundary_layout_propagated",
    lambda module, _target: propagate_function_boundary_layouts(module),
))
register_stage(Stage(
    "post-function-boundary-pack-propagation",
    "boundary_layout_propagated",
    "boundary_layout_cleaned",
    lambda module, target: post_function_boundary_pack_propagation(
        module, target),
))
register_stage(Stage(
    "decompose-paged-attention",
    "stats_threaded",
    "attention_decomposed",
    _decompose_paged_attention,
))
register_stage(Stage(
    "form-matmul-norm-stats-combine",
    "attention_decomposed",
    "stats_combined",
    lambda module, _target: form_matmul_norm_stats_combine(module),
    compatible_input_stages=frozenset({"stats_threaded"}),
))
register_stage(Stage(
    "remove-unused-functions",
    "stats_combined",
    "unused_functions_removed",
    lambda module, _target: remove_unused_functions(module),
))
register_stage(Stage(
    "propose-distribution",
    "unused_functions_removed",
    "distribution_candidates",
    _propose_distribution,
    selection_point=True,
    compatible_input_stages=frozenset({"stats_combined"}),
))
register_stage(Stage(
    "auto-distributed",
    "distribution_candidates",
    "distributed",
    _auto_distribute,
    compatible_input_stages=frozenset({
        "stats_combined",
        "unused_functions_removed",
    }),
))
register_stage(Stage(
    "fold-materialized-packed-qkv-combine",
    "distributed",
    "qkv_combine_folded",
    lambda module, _target:
        fold_materialized_packed_qkv_parallel_linear_combine(module),
    compatible_input_stages=frozenset({"distribution_candidates"}),
))
register_stage(Stage(
    "lower-packed-qkv-combine",
    "qkv_combine_folded",
    "qkv_combine_lowered",
    lambda module, _target: lower_packed_qkv_parallel_linear_combine(module),
))
register_stage(Stage(
    "sink-norm-stats-boxing",
    "qkv_combine_lowered",
    "norm_stats_boxing_sunk",
    lambda module, _target: sink_norm_stats_boxing_across_function_boundaries(module),
))
register_stage(Stage(
    "propagate-post-auto-distributed-function-boundary-layouts",
    "norm_stats_boxing_sunk",
    "distributed_boundary_layout_propagated",
    lambda module, target: propagate_post_auto_distributed_function_boundary_layouts(
        module,
        target.distributed_reshard_realization_policy(),
    ),
))
register_stage(Stage(
    "finalize-norm-stats-bindings",
    "distributed_boundary_layout_propagated",
    "norm_bindings_finalized",
    lambda module, _target: finalize_norm_stats_bindings(module),
    compatible_input_stages=frozenset({"norm_stats_boxing_sunk"}),
))
register_stage(Stage(
    "sink-finalized-norm-stats-boxing",
    "norm_bindings_finalized",
    "finalized_norm_stats_boxing_sunk",
    lambda module, _target: sink_norm_stats_boxing_across_function_boundaries(module),
))
register_stage(Stage(
    "lower-matmul-norm-stats-combine",
    "finalized_norm_stats_boxing_sunk",
    "matmul_norm_stats_lowered",
    lambda module, _target: lower_matmul_norm_stats_combine(module),
    compatible_input_stages=frozenset({"norm_bindings_finalized"}),
))
register_stage(Stage(
    "lower-vectorization-contracts",
    "matmul_norm_stats_lowered",
    "vector_contracts_lowered",
    _lower_vectorization_contracts,
    compatible_input_stages=frozenset({"norm_bindings_finalized"}),
))
register_stage(Stage(
    "fuse-norm-stats-apply",
    "vector_contracts_lowered",
    "fused_norm",
    lambda module, _target: fuse_norm_stats_apply(module),
    compatible_input_stages=frozenset({
        "matmul_norm_stats_lowered",
        # Resume checkpoints emitted before vector-contract lowering became
        # an explicit pass.
        "norm_bindings_finalized",
    }),
))
register_stage(Stage(
    "lower-tuple-boxing",
    "gather_reduce_qkv_fused",
    "tuple_boxing_lowered",
    lambda module, _target: lower_tuple_boxing(module),
    compatible_input_stages=frozenset({"frozen_constants"}),
))
register_stage(Stage(
    "propose-tir",
    "tuple_boxing_lowered",
    "selected_tir_variants",
    _propose_tir,
    selection_point=True,
    compatible_input_stages=frozenset({"frozen_constants", "gather_reduce_qkv_fused"}),
))
register_stage(Stage(
    "constant-cse",
    "fused_norm",
    "canonical_constants",
    _constant_cse,
))
register_stage(Stage(
    "lift-constant-parameters",
    "canonical_constants",
    "constant_parameters_lifted",
    lambda module, _target: lift_constant_parameter_expressions(module),
))
register_stage(Stage("freeze-constants", "constant_parameters_lifted", "frozen_constants", _freeze_constants,
                     compatible_input_stages=frozenset({"canonical_constants"})))
register_stage(Stage(
    "fuse-distributed-ops",
    "frozen_constants",
    "gather_reduce_qkv_fused",
    lambda module, _target: fuse_distributed_ops(module),
    compatible_input_stages=frozenset({"gather_reduce_add_norm_apply_fused", "gather_reduce_norm_apply_fused"}),
))
register_stage(Stage("lower-tir", "selected_tir_variants", "selected_tir", _lower_to_tir, output_dialect="semantic_tir"))
register_stage(Stage(
    "canonicalize-packed-qkv-weights",
    "selected_tir",
    "canonicalized_tir",
    _canonicalize_packed_qkv_weights,
    output_dialect="semantic_tir",
))
register_stage(Stage(
    "propose-microkernels",
    "canonicalized_tir",
    "microkernel_candidates",
    _propose_microkernels,
    output_dialect="semantic_tir",
    selection_point=True,
))
register_stage(Stage(
    "select-microkernels",
    "microkernel_candidates",
    "selected_microkernels",
    _select_microkernels,
    output_dialect="semantic_tir",
))
register_stage(Stage(
    "finalize-tir-package",
    "selected_microkernels",
    "packaged_tir",
    _finalize_tir_package,
    output_dialect="semantic_tir",
))
register_stage(Stage(
    "plan-function-memory",
    "packaged_tir",
    "memory_placed_tir",
    _plan_function_memory,
    output_dialect="semantic_tir",
))
register_stage(Stage(
    "bufferize",
    "memory_placed_tir",
    "allocated_tir",
    _bufferize,
    output_dialect="bufferized_tir",
    # Load pre-split selected_tir checkpoints produced by FlagMega v1. The
    # canonical successor remains canonicalize-packed-qkv-weights because
    # ``next_stage`` prefers exact input stages over compatibility edges.
    compatible_input_stages=frozenset({"packaged_tir", "selected_tir"}),
))
register_stage(Stage(
    "materialize-execution-functions",
    "allocated_tir",
    "scheduled_tir",
    _materialize_execution_functions,
    output_dialect="bufferized_tir",
))
register_stage(Stage(
    "plan-memory-synchronization",
    "scheduled_tir",
    "synchronized_tir",
    _plan_memory_synchronization,
    output_dialect="bufferized_tir",
))
register_stage(Stage(
    "lower-transfer-pipeline-regions",
    "synchronized_tir",
    "bufferized_tir",
    _lower_transfer_pipeline_regions,
    output_dialect="bufferized_tir",
))
