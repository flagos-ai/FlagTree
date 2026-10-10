"""Measured BI-V150 batch-one schedule."""

from dataclasses import replace
import os

from triton.flagmega.targets import IluvatarBiV150Target
from triton.flagmega.targets.iluvatar_bi_v150 import corex_triton_implementation_model



def create_target(*, glu_reduction_group=32):
    if glu_reduction_group not in {32, 64, 128}:
        raise ValueError("The reviewed GLU reduction experiments use 32/64/128 elements")
    # Full-model codegen shows that sixteen producer warps halve SME load
    # instructions again without increasing Shared or visible spills; keep
    # smaller geometries available for controlled A/B runs through the
    # environment override.
    qkv_producer_warps = int(os.environ.get("FLAGMEGA_QKV_PRODUCER_WARPS", "16"))
    if qkv_producer_warps not in {2, 4, 8, 16}:
        raise ValueError("FLAGMEGA_QKV_PRODUCER_WARPS must be 2, 4, 8, or 16")
    qkv_compute_warps = int(os.environ.get("FLAGMEGA_QKV_COMPUTE_WARPS", "32"))
    if qkv_compute_warps not in {8, 16, 32}:
        raise ValueError("FLAGMEGA_QKV_COMPUTE_WARPS must be 8, 16, or 32")
    model = corex_triton_implementation_model()

    def configure(implementation):
        parameters = dict(implementation.parameters)
        if implementation.contract.get("epilogue") == "residual_norm_stats":
            parameters["reduction_unroll"] = 8
            if implementation.variant == "packed_k_major_gemv_norm_stats":
                # The down projection covers K=6144. A 16-element K tile
                # leaves most BI-V150 lanes idle and serializes 384 reductions.
                parameters["block_k"] = 256
                parameters["tile_n"] = 64
        if implementation.family == "gather_reduce_norm_apply":
            parameters["reduction_width"] = 128
        if implementation.family == "dense_matmul_glu" and "reduction_group" in parameters:
            parameters["reduction_group"] = glu_reduction_group
            parameters["block_k"] = 64
            parameters["compute_num_warps"] = 32
        if implementation.family == "elementwise" and "elements_per_program" in parameters:
            parameters["elements_per_program"] = 256
        # One packed K group keeps the final GEMV reduction within a warp.
        # This avoids shared-memory barriers in reusable device functions.
        if (implementation.family == "qkv_parallel_linear"
                and implementation.variant == "packed_fused_gemv"):
            parameters["block_k"] = 64
            parameters["compute_num_warps"] = 32
            parameters["tile_n"] = 512
        if (implementation.family == "dense_matmul"
                and implementation.variant in {"packed_k_major_gemv", "split_k_n_packed_k_major_gemv"}):
            parameters["block_k"] = 256
            parameters["tile_n"] = 128
        if implementation.variant == "packed_pointer_smem_pipeline":
            parameters["producer_registers"] = 24
            # The SME source encoding derives its second warp axis from this
            # contract. Keep non-default geometries explicit for A/B runs.
            parameters["producer_warps"] = qkv_producer_warps
            parameters["consumer_warps"] = 8
            # The pointer QKV renderer uses this as the default WS consumer
            # partition size; its consumer_warps field is an ABI contract but
            # is not a warp-specialize launch control for this variant.
            parameters["compute_num_warps"] = qkv_compute_warps
        elif implementation.family in {"dense_matmul", "dense_matmul_glu"}:
            parameters["compute_num_warps"] = 8
        return replace(implementation, parameters=parameters)

    implementations = tuple(configure(value) for value in model.implementations)
    preferences = dict(model.preferences)
    # The async pointer pipeline is an explicit optimization target for this
    # tutorial.  Keep it out of the base BI-V150 target's default preference,
    # but expose it here so select_tir can actually materialize the WS path.
    async_qkv = "tir.qkv_parallel_linear.packed_gemv_async_smem_pipeline"
    qkv_preferences = tuple(preferences.get("qkv_parallel_linear", ()))
    if async_qkv not in {value.id for value in implementations}:
        raise RuntimeError("The BI-V150 async QKV implementation is unavailable")
    preferences["qkv_parallel_linear"] = (
        async_qkv,
        *(candidate for candidate in qkv_preferences if candidate != async_qkv),
    )
    # Keep experimental attention variants opt-in until the full-model
    # graph/chat contract has a complete end-to-end measurement across prompt
    # shapes.
    # Other targets and ordinary BI-V150 runs keep the stable portable
    # preference order unchanged.
    attention = preferences.get("paged_attention_partial", ())
    attention_variant = os.environ.get("FLAGMEGA_ATTENTION_LAYOUT")
    attention_id = (
        f"tir.paged_attention_partial.{attention_variant}"
        if attention_variant else None
    )
    if attention_id in attention:
        preferences["paged_attention_partial"] = (attention_id,) + tuple(
            candidate for candidate in attention if candidate != attention_id
        )
    return IluvatarBiV150Target(
        triton_implementation_model=replace(
            model, implementations=implementations, preferences=preferences
        )
    )
