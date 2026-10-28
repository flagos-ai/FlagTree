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
    # Match the C++ launch: sixteen physical 64-lane warps per CTA. The
    # COREX templates preserve the previous FP32 reduction trees explicitly.
    qkv_compute_warps = int(os.environ.get("FLAGMEGA_QKV_COMPUTE_WARPS", "16"))
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
                # block_k=128 rather than 256: measured, see the dense_matmul
                # note below for the numbers and for the five mechanisms
                # ruled out (including both register-spill variants).
                parameters["block_k"] = 128
                parameters["tile_n"] = 64
        if implementation.family == "gather_reduce_norm_apply":
            parameters["reduction_width"] = 128
        if implementation.family == "dense_matmul_glu" and "reduction_group" in parameters:
            parameters["reduction_group"] = glu_reduction_group
            parameters["block_k"] = 64
            parameters["compute_num_warps"] = 16
        # The selected GLU variant has no `reduction_group`, so it never
        # reached the branch above and kept the catalog's block_k=128. Exactly
        # one site uses this variant (mlp_gate_up), so this is a per-site tile.
        #
        # block_k=32, re-swept on the current base: 9.502 ms/token, against
        # (256,32) 9.521, (128,64) 9.703, (256,64) 9.889, (64,64) 9.976,
        # (512,32) 10.477, (128,128) 10.523, (128,16) 11.113, (64,32) 11.542.
        # An interior optimum -- every direction degrades.
        #
        # NOTE this is not where the first sweep landed. On the original base
        # block_k=64 won (12.038 vs 32's 12.152); after the dense_matmul,
        # template and logits changes, 32 wins by 0.200. Tile optima here are
        # NOT independent of the surrounding sites -- the same non-additivity
        # that makes keep-only deltas shift when neighbours change. Re-sweep
        # after any structural change rather than trusting a settled number.
        # Measured, not explained; see the dense_matmul note below.
        if (implementation.family == "dense_matmul_glu"
                and implementation.variant == "packed_k_major_gemv"):
            parameters["block_k"] = 32
        if implementation.family == "elementwise" and "elements_per_program" in parameters:
            parameters["elements_per_program"] = 256
        # One packed K group keeps the final GEMV reduction within a warp.
        # This avoids shared-memory barriers in reusable device functions.
        if (implementation.family == "qkv_parallel_linear"
                and implementation.variant == "packed_fused_gemv"):
            parameters["block_k"] = 64
            parameters["compute_num_warps"] = 16
            parameters["tile_n"] = 512
        # These two dense_matmul variants want OPPOSITE tiles, so they must be
        # set per variant. Tuning them jointly (one tile for both) averages
        # two different optima: it cost 0.18 ms and left the three
        # dense_matmul sites at 1.5-1.7x behind the cpp backend's per-site
        # bandwidth while the separately-tuned GLU reached 1.13x.
        #
        # The tiles are empirical and the mechanism is NOT established; five
        # candidates are eliminated by measurement (ITERATION.md Trial 95.5,
        # 97b.6). The tuned build executes 3.58x MORE inner loop steps than
        # the old one and is still faster, so the win is in per-step cost,
        # not step count. Ruled out: vector-register footprint
        # (vgpr_spill_count is 0 in every build measured), block_k ==
        # warpSize, fewer steps, load vectorization width, and scalar spill
        # count -- sgpr spills really do fall 292 -> 159 across this retune,
        # but the 2nd-fastest build carries the MOST at 335 while a build
        # 1.5 ms slower carries 162, so they co-vary with the tile rather
        # than driving the time. Do not re-derive these numbers from any of
        # those models; re-measure.
        if (implementation.family == "dense_matmul"
                and implementation.variant == "split_k_n_packed_k_major_gemv"):
            # (tile_n=128, block_k=256): 10.514 ms/token, against (64,128)
            # 10.693, (128,128) 10.526, (128,512) 10.514 (tie), (256,256)
            # 10.625, (256,128) 10.832, (64,256) 10.713. tile_n is the live
            # knob here; block_k is flat from 256 up.
            parameters["block_k"] = 256
            parameters["tile_n"] = 128
        if (implementation.family == "dense_matmul"
                and implementation.variant == "packed_k_major_gemv"):
            # (tile_n=128, block_k=64). Exactly one site in this workload uses
            # this variant -- the logits/lm_head GEMV (N=9496, K=2048) -- so
            # this is a per-site tile, not a shared one.
            #
            # It was (64,128), carried over from a sweep where the same
            # variant still served several sites. Re-attributing per site with
            # MATCHED stubbing (the same keep pattern applied to both
            # backends; see ITERATION.md 97b.8) showed logits was 1.37x behind
            # the cpp backend and accounted for 61% of the whole remaining
            # gap, while every other site was within 1.09x. Re-sweeping this
            # site alone: (128,64) 9.674 ms/token, (256,64) 9.849,
            # (64,128) 10.009, (256,128) 10.140, (128,32) 10.379,
            # (128,128) 10.515, (64,256) 11.116, (128,16) 11.618.
            # (192,64) is not expressible -- tiles must be powers of two.
            #
            # Hypothesis for why a wider N tile wins here and not elsewhere,
            # NOT demonstrated: cpp's logits kernel runs a `dense_stream` loop
            # with 4 concurrent streams per thread while this one is a single
            # dependent chain, and this card is latency-bound on in-flight
            # load count. N=9496 is 4.6x wider than any other site, so there
            # is the most latency to hide. The probe that would have tested
            # this directly was void by its own validity gate (276.7 GB/s
            # against the real site's 343).
            parameters["block_k"] = 64
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
    # That WS path currently deadlocks at runtime (ITERATION.md Trials 80-82:
    # the `tle.pipe` + `warp_specialize` rendezvous is a primitive-level
    # deadlock, not a megakernel interaction), so preferring it first makes
    # this profile uncompilable. Keep the experiment opt-in until that defect
    # is fixed; the shipping path is the non-p/c physical-order template.
    if os.environ.get("FLAGMEGA_QKV_ASYNC") == "1":
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
