"""AMD RDNA recipe for Qwen3-1.7B decode (gfx1201 / R9700 and gfx1100 / W7900).

The FlagMega distribution recipe is mesh-topology based (an 8x16 "bb" block
grid), so it is vendor- AND gfx-agnostic: this module reuses the measured
distribution CHOICES and the residual/norm layout helpers from the sibling
nvidia-h800 tutorial, and the one recipe serves both RDNA3 (gfx1100) and RDNA4
(gfx1201). Only three things are AMD-specific:

  1. ``create_target(arch=...)`` wraps ``AmdGfx1100Target`` / ``AmdGfx1201Target``
     instead of ``NvidiaSm90Target`` (same batch-1 kernel-parameter tweaks --
     tile/reduction sizes, not NV hardware features). The gfx target machine is
     the only difference between the two RDNA parts.
  2. ``residual_layout="sharded-casts"`` + ``norm_layout="replicated-residual"``
     keep the residual/norm shards full-mesh contiguous. The norm layout is the
     load-bearing one: under the default auto-distribution the solver gives the
     fused ``gather_reduce_add_norm_apply`` a block-cyclic partial input, which
     its renderer rejects ("Cross-layout partial access requires contiguous
     source shards"). The residual pin alone is no longer required, but lifting
     it changes the reduction order (logits not bitwise) and removes no grid
     barrier.
  3. The H800 TMA kernel recipe (``local_optimizations.kernel_recipe.select_tir``)
     is SKIPPED. RDNA has no TMA/tensor-descriptor, so the default TIR selection
     picks the portable ``packed_k_major_gemv`` variants already supported here.
     The exceptions are the MLP gate+up projection (``_GLU_PROJECTION``, a
     64-row tile) and the paged-attention partial (``_ATTENTION_PARTIAL``).

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
from triton.flagmega.targets.amd_gfx1100 import AmdGfx1100Target
from triton.flagmega.targets.amd_gfx1201 import AmdGfx1201Target
from triton.flagmega.targets.portable_triton_implementations import portable_triton_implementation_model

# The distribution recipe below is mesh-topology based, so one tutorial serves
# both RDNA parts; only the gfx target machine differs.
_TARGETS = {"gfx1100": AmdGfx1100Target, "gfx1201": AmdGfx1201Target}

from local_optimizations.distribution_recipe import (  # noqa: E402  (sibling tutorial)
    CHOICES,
    replicated_norm_choices,
    sharded_residual_norm_choices,
)


def create_target(*, arch: str = "gfx1201", glu_reduction_group: int = 32):
    """An RDNA target (``gfx1201``/R9700 or ``gfx1100``/W7900) with the reviewed
    batch-1 kernel-parameter tweaks.

    The distribution recipe and the tile/reduction tweaks are mesh-topology and
    software choices shared across RDNA3 and RDNA4; only the gfx target machine
    differs. The implementation model is reused from NVIDIA (bf16 kernels are
    portable).
    """
    if arch not in _TARGETS:
        raise ValueError(f"arch must be one of {sorted(_TARGETS)}, got {arch!r}")
    if glu_reduction_group not in {32, 64, 128}:
        raise ValueError("The reviewed GLU reduction experiments use 32/64/128 elements")
    model = portable_triton_implementation_model()

    def configure(implementation):
        parameters = dict(implementation.parameters)
        if implementation.contract.get("epilogue") == "residual_norm_stats":
            parameters["reduction_unroll"] = 8
        if implementation.family == "gather_reduce_norm_apply":
            parameters["reduction_width"] = 128
        if implementation.family == "dense_matmul_glu" and "reduction_group" in parameters:
            parameters["reduction_group"] = glu_reduction_group
        if implementation.id == _GLU_PROJECTION:
            parameters["block_k"] = _GLU_BLOCK_K
        if implementation.family == "elementwise" and "elements_per_program" in parameters:
            parameters["elements_per_program"] = 256
        return replace(implementation, parameters=parameters)

    return _TARGETS[arch](triton_implementation_model=replace(
        model, implementations=tuple(configure(value) for value in model.implementations)))


# Split-K GEMV points: splitting the reduction over y (partial_p_sum) AND sharding
# the output over x (s_c/s_bc_h1) runs the GEMV on all 128 CTAs instead of the 16
# that own a single x-row, flooding the CUs. On R9700 this is ~3x faster for the
# 152K-vocab lm_head and ~1.3x for the MLP down-projection -- the default
# output-only shards under-parallelize these big reductions.
_SPLIT_K_POINTS = (
    "distribution.logits.vectorized.compute",
    # Real-importer (VLLM_INDUCTOR_LEVEL3) names the wide per-layer projections:
    "distribution.decode_layer_after_attention.projection_wide.vectorized.compute",
    "distribution.decode_layer_output.projection_wide.vectorized.compute",
    # Synthetic import_qwen3_model name (harmless no-op on the real graph):
    "distribution.decode_layer_mlp_down.vectorized.compute",
)


def _split_k_choices(points):
    fixed = {}
    for point_id in _SPLIT_K_POINTS:
        point = points.get(point_id)
        if point is None:
            continue
        for cand in point.candidates:
            cid = cand.id
            if "partial_p_sum" in cid and ("_s_c_h1" in cid or "_s_bc_h1" in cid):
                fixed[point_id.removeprefix("distribution.")] = cid
                break
    return fixed


def select_distribution(module, target, *, residual_layout="sharded-casts",
                        norm_layout="replicated-residual", exclusive_axes=(0, 1),
                        split_k_logits=True):
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
    if split_k_logits:
        fixed.update(_split_k_choices(points))
    result = solve_search_graph(graph, fixed_selections=fixed)
    _, records = distribution_selection_state(result, policy=target.distribution_policy.identity)
    return override_plan(
        module,
        [(record.point_id, record.candidate_id) for record in records],
        rationale=("RDNA batch-1 Qwen3-1.7B; sharded-casts residual + replicated-residual norm; "
                   "RDNA portable GEMV (no TMA)"),
    )


# On AMD decode_t16 renders the per-token-row online softmax
# (platforms/amd/decode_t16), which beats the shared-state decode_t32 from 1K up
# without raising the megakernel's VGPRs. On gfx1100 the long-context partial is
# latency-bound, so platforms/amd/gfx1100 also loads branch-free clamped tiles
# and the next tile's block ids with this tile's K/V (W7900 8K partial 65 -> 50
# us/layer). Per-row 32-token tiles raise VGPRs; t128/t256 spill.
_ATTENTION_PARTIAL = "tir.paged_attention_partial.decode_t16"

# The default gate+up tile (tile_n 16) sweeps each CTA's 48 GLU rows in three
# passes over K, splitting every contiguous 1.5 KB weight run into three reads
# at different times. W7900 loses DRAM locality to that (71% of achievable
# bandwidth); one 64-row tile with 32-wide K steps streams at the ceiling
# without raising the megakernel's peak VGPRs (gate+up 105 -> 72 us/layer).
# R9700 already streams gate+up at ~98% and is neutral.
_GLU_PROJECTION = "tir.dense_matmul_glu.packed_k_major_gemv_tn64"
_GLU_BLOCK_K = 32


def _pin_implementation(module, implementation_id):
    choices = [
        (point.id, implementation_id)
        for point in module.selection_points
        if any(candidate.id == implementation_id for candidate in point.candidates)
    ]
    if not choices:
        raise ValueError(f"{implementation_id} is not offered at any selection point")
    return override_plan(module, choices, rationale=f"RDNA recipe: {implementation_id}")


def recipe_compile(module, compiler, target):
    """Compile an imported Qwen3-1.7B module to bufferized_tir with the AMD recipe.

    Distribution is pinned by the recipe. TIR selection is left to the default
    policy (the RDNA-portable packed_k_major_gemv kernels) except for the MLP
    gate+up projection (``_GLU_PROJECTION``, pinned when TIR is lowered) and the
    paged-attention partial (``_ATTENTION_PARTIAL``, pinned among microkernels).
    """
    module = compiler.compile(module, stop_after="propose-distribution").module
    plan = select_distribution(module, target)
    module = compiler.run_stage(module, "auto-distributed", plan=plan).module
    module = compiler.compile(module, stop_after="propose-tir").module
    module = compiler.run_stage(
        module, "lower-tir", plan=_pin_implementation(module, _GLU_PROJECTION)
    ).module
    module = compiler.compile(module, stop_after="propose-microkernels").module
    module = compiler.run_stage(
        module, "select-microkernels", plan=_pin_implementation(module, _ATTENTION_PARTIAL)
    ).module
    return compiler.compile(module).module
