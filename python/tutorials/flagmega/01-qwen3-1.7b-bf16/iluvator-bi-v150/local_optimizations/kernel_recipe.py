"""BI-V150 transport choices over COREX-compatible Triton kernels."""

import os

from triton.flagmega.selection import override_plan


def select_tir(module, *, residual_kernel="direct"):
    if residual_kernel not in {"direct", "staged", "async"}:
        raise ValueError(f"Unknown residual transport {residual_kernel}")
    choices = []
    residual_points = 0
    # tile_n=16 was the inherited SM90 default (its higher-priority TMA
    # candidates don't survive COREX's portable_triton filter, silently
    # exposing tn16 as a de facto choice, not a measured one). tile_n=16
    # means mlp_gate_up's packed-layout swizzle address arithmetic
    # (`(k//16)*stride + (n//8)*128 + ...`, div/mod-heavy per element) is
    # paid every 16 output columns instead of every 64; tn64 amortizes it
    # over 4x more work per address computed. See perf-iteration/
    # ITERATION.md Trial 27 (225x-slower-than-vLLM baseline) and the trial
    # documenting this change for the measured effect.
    glu = "tir.dense_matmul_glu.packed_k_major_gemv_tn64"
    base = "tir.dense_matmul.packed_k_major_gemv_norm_stats"
    attention_variant = os.environ.get("FLAGMEGA_ATTENTION_LAYOUT")
    attention_layout = (
        f"tir.paged_attention_partial.{attention_variant}"
        if attention_variant else None
    )
    # Same tile_n=16-by-inherited-ordering-accident issue as `glu` above,
    # for qkv_projection (Trial 30: 11.7x slower than vLLM). No tn32/tn64
    # sibling existed for qkv_parallel_linear/packed_fused_gemv before
    # this trial added `..._tn64`; measure its effect the same way before
    # relying on it further.
    qkv = "tir.qkv_parallel_linear.packed_fused_gemv_tn64"
    async_qkv = "tir.qkv_parallel_linear.packed_gemv_async_smem_pipeline"
    # projection_wide (Trial 63): the site was de facto on the small
    # tiles via inherited ordering; force the largest registered packed
    # GEMV tile (tn64_bk256) the same way the glu override works.
    proj = "tir.dense_matmul.packed_k_major_gemv_tn64_bk256"
    for point in module.selection_points:
        # Microkernels are materialized as ``tir_microkernel`` points after
        # semantic TIR lowering; paged attention lives there rather than in a
        # plain ``tir`` point.  Keep both kinds eligible for the explicit
        # layout experiment.
        if point.kind not in {"tir", "tir_microkernel"}:
            continue
        ids = {candidate.id for candidate in point.candidates}
        if glu in ids:
            choices.append((point.id, glu))
        if async_qkv in ids:
            choices.append((point.id, async_qkv))
        elif qkv in ids:
            choices.append((point.id, qkv))
        if proj in ids:
            choices.append((point.id, proj))
        if attention_layout is not None and attention_layout in ids:
            choices.append((point.id, attention_layout))
        if any(c.parameters.get("epilogue") == "residual_norm_stats" for c in point.candidates):
            if base not in ids:
                raise ValueError(f"COREX residual GEMV is not legal at {point.id}")
            choices.append((point.id, base))
            residual_points += 1
    if residual_kernel != "direct" and not residual_points:
        raise ValueError("Requested residual transport has no applicable GEMV points")
    return override_plan(module, choices, rationale=f"COREX BI-V150 GEMV transport={residual_kernel}")
