"""AMD RDNA (Radeon AI PRO R9700 / gfx1201) recipe for Qwen3-1.7B decode.

The FlagMega distribution recipe is mesh-topology based (an 8x16 "bb" block
grid), so it is vendor-agnostic: this module reuses the measured distribution
CHOICES and the residual/norm layout helpers from the sibling nvidia-h800
tutorial. Only three things are AMD-specific:

  1. ``create_target`` wraps ``AmdGfx1201Target`` instead of ``NvidiaSm90Target``
     (same batch-1 kernel-parameter tweaks -- tile/reduction sizes, not NV
     hardware features).
  2. ``residual_layout="sharded-casts"`` + ``norm_layout="replicated-residual"``
     keep the residual/norm shards full-mesh contiguous, which avoids the
     gather-reduce owner-group codegen path that the default auto-distribution
     hits at realistic (hidden=2048) shapes.
  3. The H800 TMA kernel recipe (``local_optimizations.kernel_recipe.select_tir``)
     is SKIPPED. RDNA has no TMA/tensor-descriptor, so the default TIR selection
     picks the portable ``packed_k_major_gemv`` variants already supported here.

Put this directory on PYTHONPATH (the sibling nvidia-h800 directory is added
automatically) and call ``recipe_compile`` on an imported Qwen3-1.7B module.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

_NVIDIA_TUTORIAL = Path(__file__).resolve().parent.parent / "nvidia-h800"
if str(_NVIDIA_TUTORIAL) not in sys.path:
    sys.path.insert(0, str(_NVIDIA_TUTORIAL))

from triton.flagmega.passes.auto_distributed import (
    DistributedCandidateProviderRegistry,
    build_search_graph,
    solve_search_graph,
)
from triton.flagmega.passes.auto_distributed.materializer import distribution_selection_state
from triton.flagmega.selection import override_plan
from triton.flagmega.targets.amd_gfx1201 import AmdGfx1201Target
from triton.flagmega.targets.nvidia import sm90_triton_implementation_model

from local_optimizations.distribution_recipe import (  # noqa: E402  (sibling tutorial)
    CHOICES,
    replicated_norm_choices,
    sharded_residual_norm_choices,
)


def create_target(*, glu_reduction_group: int = 32) -> AmdGfx1201Target:
    """An R9700/gfx1201 target with the reviewed batch-1 kernel-parameter tweaks.

    The implementation model is reused from NVIDIA (bf16 kernels are portable);
    only the reduction/tile parameters are specialized, exactly as the H800
    tutorial does -- these are software tile sizes, not NV hardware features.
    """
    if glu_reduction_group not in {32, 64, 128}:
        raise ValueError("The reviewed GLU reduction experiments use 32/64/128 elements")
    model = sm90_triton_implementation_model()

    def configure(implementation):
        parameters = dict(implementation.parameters)
        if implementation.contract.get("epilogue") == "residual_norm_stats":
            parameters["reduction_unroll"] = 8
        if implementation.family == "gather_reduce_norm_apply":
            parameters["reduction_width"] = 128
        if implementation.family == "dense_matmul_glu" and "reduction_group" in parameters:
            parameters["reduction_group"] = glu_reduction_group
        if implementation.family == "elementwise" and "elements_per_program" in parameters:
            parameters["elements_per_program"] = 256
        return replace(implementation, parameters=parameters)

    return AmdGfx1201Target(triton_implementation_model=replace(
        model, implementations=tuple(configure(value) for value in model.implementations)))


def select_distribution(module, target, *, residual_layout="sharded-casts",
                        norm_layout="replicated-residual", exclusive_axes=(0, 1)):
    """Apply the (mesh-agnostic) Qwen3-1.7B distribution recipe for an RDNA mesh."""
    points = {point.id: point for point in module.selection_points if point.kind == "distribution"}
    choices = []
    for point_id, candidate_id in CHOICES:
        if point_id not in points:
            raise ValueError(f"The workload changed: missing distribution point {point_id}")
        if not any(candidate.id == candidate_id for candidate in points[point_id].candidates):
            raise ValueError(f"Recipe candidate is no longer legal: {candidate_id}")
        choices.append((point_id, candidate_id))
    registry = DistributedCandidateProviderRegistry()
    target.register_auto_distributed_candidate_providers(registry)
    placements = target.distributed_placements(module)
    if len(placements) != 1:
        raise ValueError("Choose a single mesh before applying this workload recipe")
    graph = build_search_graph(module, placements[0], registry,
                               target.distributed_reshard_realization_policy(),
                               target.distributed_reshard_cost_model(),
                               target.distributed_operation_cost_model())
    fixed = {point.removeprefix("distribution."): candidate for point, candidate in choices}
    if residual_layout in {"sharded", "sharded-casts"}:
        fixed.update(sharded_residual_norm_choices(graph, module, shard_casts=residual_layout == "sharded-casts"))
    if norm_layout in {"replicated", "replicated-residual", "exclusive", "exclusive-residual"}:
        fixed.update(replicated_norm_choices(
            graph, module,
            residual_only=norm_layout in {"replicated-residual", "exclusive-residual"},
            exclusive=norm_layout in {"exclusive", "exclusive-residual"},
            exclusive_axes=exclusive_axes,
        ))
    result = solve_search_graph(graph, fixed_selections=fixed)
    _, records = distribution_selection_state(result, policy=target.distribution_policy.identity)
    return override_plan(
        module,
        [(record.point_id, record.candidate_id) for record in records],
        rationale=("R9700 batch-1 Qwen3-1.7B; sharded-casts residual + replicated-residual norm; "
                   "RDNA portable GEMV (no TMA)"),
    )


def recipe_compile(module, compiler, target):
    """Compile an imported Qwen3-1.7B module to bufferized_tir with the AMD recipe.

    Distribution is pinned by the recipe; TIR selection is left to the default
    policy, which picks the RDNA-portable packed_k_major_gemv kernels.
    """
    module = compiler.compile(module, stop_after="propose-distribution").module
    plan = select_distribution(module, target)
    module = compiler.run_stage(module, "auto-distributed", plan=plan).module
    return compiler.compile(module).module
