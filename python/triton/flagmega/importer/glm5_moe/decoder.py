# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""GLM-5.3-Flash decoder construction: nope MLA, noaux_tc routing, clamped SwiGLU MoE."""

from triton.flagmega.ir import F, tensor_type


def linear(value, weight, *, weight_scale=None, config=None, name=None, output_dtype=None):
    if weight_scale is None:
        return F.math.matmul(value, weight, transpose_b=True, output_data_type=output_dtype, name=name)
    if config is None:
        raise ValueError("GLM block-scaled projection requires its weight-block configuration.")
    result = F.math.block_scaled_matmul(
        value, weight, weight_scale,
        weight_block_n=config.weight_block_n, weight_block_k=config.weight_block_k, name=name)
    if output_dtype is not None and str(output_dtype) != str(result.type.dtype.value):
        result = F.tensors.cast(result, output_dtype, name=f"{name}_cast")
    return result


def projection(value, weights, key, config, *, prefix, output_dtype=None):
    weight = weights[key]
    scale = weights.get(key + "_scale_inv")
    return linear(value, weight, weight_scale=scale, config=config,
                  output_dtype=output_dtype, name=f"{prefix}_{key.replace('.', '_')}_apply")


def rms_norm(value, weight, epsilon, *, name, output_dtype="bfloat16"):
    # GLM applies the plain weight (no Gemma-style +1) in FP32.
    wide_weight = F.tensors.cast(weight, "float32")
    zero = F.builtin.splat_const(wide_weight.type, 0.0)
    stats = F.nn.norm_stats(value, axis=-1, use_mean=False)
    return F.nn.norm_apply(value, stats, wide_weight, zero, axis=-1, epsilon=epsilon, use_mean=False,
                           round_before_scale=False, output_dtype=output_dtype, name=name)


def gated_rms_norm(value, gate, weight, epsilon, *, name, activation="sigmoid",
                   output_dtype="bfloat16"):
    """``FusedRMSNormGated``: one FP32 interior, one rounding at ``output_dtype``.

    The gate enters raw. Splitting this into norm_apply/sigmoid/mul would round
    twice against the reference: the normalized value to the input dtype, and
    the activation to the gate dtype.
    """
    wide_weight = F.tensors.cast(weight, "float32")
    zero = F.builtin.splat_const(wide_weight.type, 0.0)
    stats = F.nn.norm_stats(value, axis=-1, use_mean=False)
    return F.nn.gated_norm_apply(value, stats, wide_weight, zero, gate, axis=-1, epsilon=epsilon,
                                 use_mean=False, activation=activation, round_before_scale=False,
                                 output_dtype=output_dtype, name=name)


def build_hyper_connection(streams, weights, config, *, stage, prefix):
    """Build one GLM-5 mHC projection and collapse its input streams.

    The reference implementation computes the coefficients from all streams,
    then consumes only the collapsed value in the attention/MLP block.  One
    ``nn.mhc_pre`` carries the whole stage: the RMS-normalized projection, the
    three coefficient groups, the Sinkhorn normalization, and the collapse.
    Expanding it as graph algebra costs one node per Sinkhorn iteration and
    direction, and pins every intermediate to memory.
    """
    stage_result = F.nn.mhc_pre(
        streams, weights[f"hc_{stage}_fn"], weights[f"hc_{stage}_base"],
        weights[f"hc_{stage}_scale"], epsilon=config.epsilon,
        sinkhorn_iterations=config.hc_sinkhorn_iters, sinkhorn_epsilon=config.hc_eps,
        name=f"{prefix}_{stage}_hyper_connection")
    return F.tensors.get_items(stage_result, 0, 1, 2, name_prefix=f"{prefix}_{stage}")


def apply_hyper_connection(value, residual, post, comb, config, *, stage, prefix):
    """Expand a block result and add the doubly-stochastic residual mix."""
    return F.nn.mhc_post(value, residual, post, comb, name=f"{prefix}_{stage}_output")


def build_mla_attention(value, state, layer_id, advance, weights, config, *, prefix,
                        dsa_state=None, dsa_layer_id=None, dsa_advance=None, topk=None,
                        indexer_kind=None):
    """Unabsorbed nope MLA over the paged KV cache.

    The compressed latent is expanded per head through ``kv_b_proj`` before the
    cache write, so the standard paged attention serves as-is; with
    ``qk_rope_head_dim == 0`` there is no rotary stage at all.
    """
    w = lambda name: weights[f"self_attn.{name}"]
    c = config
    tokens = value.type.shape[0].fixed_value
    heads = c.num_attention_heads

    q_resid = projection(value, weights, "self_attn.q_a_proj.weight", c, prefix=prefix)
    q_resid = rms_norm(q_resid, w("q_a_layernorm.weight"), c.epsilon, name=f"{prefix}_q_a_norm")
    query = projection(q_resid, weights, "self_attn.q_b_proj.weight", c, prefix=prefix)
    query = F.tensors.reshape(query, (tokens, heads, c.qk_nope_head_dim), name=f"{prefix}_query_heads")

    compressed = projection(value, weights, "self_attn.kv_a_proj_with_mqa.weight", c, prefix=prefix)
    compressed = rms_norm(compressed, w("kv_a_layernorm.weight"), c.epsilon, name=f"{prefix}_kv_a_norm")
    layout = ("seq", "head", "dim")
    packed_q = F.tensors.pack(query, 8, axis=-1)
    if "self_attn.kv_b_proj.weight" in weights:
        # Native FP8 checkpoints keep one interleaved [K,V] projection. Do
        # the exact row regrouping on the result, after the single scaled GEMM;
        # slicing the FP8 weight would require slicing its block-scale grid.
        kv = projection(compressed, weights, "self_attn.kv_b_proj.weight", c, prefix=prefix)
        span = c.qk_nope_head_dim + c.v_head_dim
        # Keep the complete head-major [K | V] source. The paired cache
        # consumer selects each field in its loads, without slice buffers.
        kv = F.tensors.reshape(kv, (tokens, heads, span), name=f"{prefix}_kv_heads")
        state = F.nn.update_paged_attention_kv_cache_pair(
            F.tensors.pack(kv, 8, axis=-1), state, layer_id, advance, layout=layout)
    else:
        # BF16 checkpoints are regrouped in the constant island and passed as
        # two ordinary projections to keep the legacy path compact.
        key = F.tensors.reshape(linear(compressed, w("k_b_proj"), name=f"{prefix}_k_b"),
                                (tokens, heads, c.qk_nope_head_dim), name=f"{prefix}_key_heads")
        val = F.tensors.reshape(linear(compressed, w("v_b_proj"), name=f"{prefix}_v_b"),
                                (tokens, heads, c.v_head_dim), name=f"{prefix}_value_heads")
        # Separate projections retain the single-field API; only V advances.
        packed_k, packed_v = (F.tensors.pack(item, 8, axis=-1) for item in (key, val))
        no_advance = F.builtin.scalar_const(tensor_type("bool", ()), False)
        state = F.nn.update_paged_attention_kv_cache(packed_k, state, layer_id, no_advance, cache_kind="key", layout=layout)
        state = F.nn.update_paged_attention_kv_cache(packed_v, state, layer_id, advance, cache_kind="value", layout=layout)
    output_topk = topk
    updated_dsa = dsa_state
    if c.indexer_types:
        if indexer_kind == "full":
            indexer_q = F.math.matmul(
                q_resid, w("indexer.wq_b.weight"), transpose_b=True,
                name=f"{prefix}_dsa_indexer_q")
            indexer_key = F.math.matmul(
                value, w("indexer.wk.weight"), transpose_b=True,
                name=f"{prefix}_dsa_indexer_key")
            indexer_gate = F.math.matmul(
                value, w("indexer.index_kpool_compress_gate"), transpose_b=True,
                name=f"{prefix}_dsa_indexer_gate")
            indexer_weights = F.math.matmul(
                value, w("indexer.weights_proj.weight"), transpose_b=True,
                name=f"{prefix}_dsa_indexer_weights")
            output_topk = F.nn.dsa_indexer(
                indexer_q, indexer_key, indexer_gate, indexer_weights, dsa_state,
                dsa_layer_id if dsa_layer_id is not None else layer_id, dsa_advance,
                w("indexer.k_norm.weight"), w("indexer.k_norm.bias"),
                w("indexer.index_kpool_compress_ape"), index_topk=c.index_topk,
                index_head_dim=c.index_head_dim, index_n_heads=c.index_n_heads,
                index_kpool=c.index_kpool, always_select_tail=c.index_kpool_always_select_tail,
                epsilon=c.epsilon, name=f"{prefix}_dsa_indexer")
            updated_dsa = dsa_state
        attended = F.nn.dsa_paged_attention(
            packed_q, output_topk, state, layer_id, scale=c.qk_nope_head_dim ** -0.5,
            layout=layout, hidden_size=c.query_size, name=f"{prefix}_dsa_attention")
    else:
        attended = F.nn.paged_attention(packed_q, state, layer_id, scale=c.qk_nope_head_dim**-0.5,
                                        layout=layout, hidden_size=c.query_size)
    context = F.tensors.reshape(F.tensors.unpack(attended, axis=-1), (tokens, c.query_size),
                                name=f"{prefix}_attention_context")
    output = projection(context, weights, "self_attn.o_proj.weight", c, prefix=prefix)
    return output, state, updated_dsa, output_topk


def build_kda_attention(value, conv_state, recurrent_state, weights, config, *, prefix):
    """GLM-5.3 KDA decode step per the published modeling.

    Mixed q/k/v through the depthwise short convolution (with SiLU), then one
    KDA recurrent step with the per-channel log-decay ``lower_bound *
    sigmoid(exp(A_log)·(g + dt_bias))`` from the published forget gate, a
    per-head sigmoid write strength, L2-normalized q/k, and a gated RMSNorm on
    the output before the out projection.
    """
    w = lambda name: weights[f"self_attn.{name}"]
    c = config
    tokens = value.type.shape[0].fixed_value
    heads = c.kda_num_heads
    dim = c.kda_head_dim

    # The three projections stay separate and head-major all the way through the
    # convolution, exactly like upstream's independent q/k/v_conv1d layers. A
    # flat [1, 3*heads*head_dim] concat would interleave the parts on one axis
    # and make the head axis unsplittable.
    projections = tuple(
        F.tensors.reshape(projection(value, weights, f"self_attn.{part}_proj.weight", c, prefix=prefix),
                          (tokens, heads, dim), name=f"{prefix}_{part}_proj_heads")
        for part in ("q", "k", "v"))
    convolved_tuple = F.nn.kda_convolution(*projections, conv_state, w("conv1d.weight"),
                                           kernel_size=c.kda_conv_kernel_size, name=f"{prefix}_conv")
    q, k, v, conv_state_out = F.tensors.get_items(convolved_tuple, 0, 1, 2, 3,
                                                  name_prefix=f"{prefix}_conv")

    # Per-channel forget gate off the RAW hidden states, exactly the published
    # Glm5NextTextForgetGate: f_a → f_b low-rank, then the FP32 log-decay
    # ``lower_bound * sigmoid(exp(A_log)·(g + dt_bias))`` per channel, which
    # kda_recurrent_core consumes as ``h *= exp(g)``. ``dt_bias_map`` and
    # ``decay_rate_map`` arrive as ready weight transforms from main's
    # constant island (exp/broadcast of A_log and dt_bias), so the runtime
    # dataflow has no weight-side broadcasts to vectorize.
    gate_a = projection(value, weights, "self_attn.f_a_proj.weight", c, prefix=prefix)
    gate_raw = projection(gate_a, weights, "self_attn.f_b_proj.weight", c, prefix=prefix)
    gate_heads = F.tensors.reshape(gate_raw, (tokens, heads, dim), name=f"{prefix}_gate_heads")
    # The raw ``b_proj`` logits, not a pre-sigmoided beta: the recurrent kernel
    # widens and sigmoids them, which is both the published
    # ``torch.sigmoid(self.b_proj(hidden_states))`` order and upstream's
    # ``use_beta_sigmoid_in_kernel=True``.
    beta_logits = projection(value, weights, "self_attn.b_proj.weight", c, prefix=prefix)

    recurrent = F.nn.kda_gated_recurrent_core(
        q, k, v, gate_heads, w("decay_rate_map"), w("dt_bias_map"), beta_logits, recurrent_state,
        lower_bound=c.kda_gate_lower_bound, epsilon=1e-6, l2norm_in_kernel=True,
        name=f"{prefix}_recurrent")
    core_out, recurrent_state_out = F.tensors.get_items(recurrent, 0, 1, name_prefix=f"{prefix}_recurrent")
    # The published module is FusedRMSNormGated(o_norm)(core_out, g), whose
    # interior is entirely FP32: the gate's sigmoid and the multiply never round.
    out_gate = F.tensors.reshape(
        projection(projection(value, weights, "self_attn.g_a_proj.weight", c, prefix=prefix),
                   weights, "self_attn.g_b_proj.weight", c, prefix=prefix),
        (tokens, heads, dim), name=f"{prefix}_out_gate_heads")
    gated = gated_rms_norm(core_out, out_gate, w("o_norm.weight"), c.epsilon,
                           activation="sigmoid", output_dtype="bfloat16",
                           name=f"{prefix}_o_norm_gated")
    output = projection(F.tensors.reshape(gated, (tokens, heads * dim), name=f"{prefix}_out_flat"),
                        weights, "self_attn.o_proj.weight", c, prefix=prefix)
    return output, conv_state_out, recurrent_state_out


def sigmoid_noaux_tc_routing(value, weights, config, *, prefix):
    """GLM/DeepSeek noaux_tc routing: select on ``sigmoid(x) + bias``, weight on ``sigmoid(x)``.

    One staged-routing stage replaces the sigmoid/add/top_k/gather/sum/div/mul
    chain; selection ties resolve to the lower expert id exactly like
    ``tensors.top_k``.
    """
    logits = linear(value, weights["mlp.gate.weight"], output_dtype="float32", name=f"{prefix}_router_logits")
    bias = F.tensors.cast(weights["mlp.gate.e_score_correction_bias"], "float32", name=f"{prefix}_router_bias_wide")
    top = F.nn.staged_routing(logits, bias, k=config.num_experts_per_tok, axis=-1, index_dtype="int32",
                              scoring="sigmoid", normalize=True, scaling=config.routed_scaling_factor,
                              name=f"{prefix}_router")
    return F.tensors.get_items(top, 0, 1, name_prefix=f"{prefix}_router")


def build_moe(value, weights, config, *, prefix):
    c = config
    tokens = value.type.shape[0].fixed_value
    router_weights, ids = sigmoid_noaux_tc_routing(value, weights, config, prefix=prefix)
    ones = F.builtin.splat_const(tensor_type("float32", (c.num_experts, 1)), 1.0)
    routed = F.nn.sparse_experts(
        value, ids, router_weights, ones, weights["mlp.experts.gate_proj"],
        weights.get("mlp.experts.gate_proj_scale_inv", ones), ones,
        weights["mlp.experts.down_proj"], weights.get("mlp.experts.down_proj_scale_inv", ones),
        ones, weights["mlp.experts.up_proj"], weights.get("mlp.experts.up_proj_scale_inv", ones),
        gate_clamp_max=c.swiglu_limit, up_clamp=c.swiglu_limit,
        weight_block_n=c.weight_block_n, weight_block_k=c.weight_block_k,
        name=f"{prefix}_routed_experts")
    unit = F.builtin.splat_const(tensor_type("float32", (1, 1)), 1.0)
    shared_ids = F.builtin.splat_const(tensor_type("int32", (tokens, 1)), 0)
    shared_units = F.builtin.splat_const(tensor_type("float32", (tokens, 1)), 1.0)
    shared_weights = {
        stage: F.tensors.reshape(weights[f"mlp.shared_experts.{stage}_proj.weight"],
                                 shape=(1, *(d.fixed_value for d in
                                             weights[f"mlp.shared_experts.{stage}_proj.weight"].type.shape)))
        for stage in ("gate", "up", "down")
    }
    shared_scales = {
        stage: (F.tensors.reshape(weights[f"mlp.shared_experts.{stage}_proj.weight_scale_inv"],
                                  shape=(1, *(d.fixed_value for d in
                                              weights[f"mlp.shared_experts.{stage}_proj.weight_scale_inv"].type.shape)))
                if f"mlp.shared_experts.{stage}_proj.weight_scale_inv" in weights else unit)
        for stage in ("gate", "up", "down")
    }
    # Always active at weight one, never part of the top-k normalization.
    shared = F.nn.sparse_experts(
        value, shared_ids, shared_units, unit, shared_weights["gate"], shared_scales["gate"], unit,
        shared_weights["down"], shared_scales["down"], unit, shared_weights["up"], shared_scales["up"],
        gate_clamp_max=c.swiglu_limit, up_clamp=c.swiglu_limit,
        weight_block_n=c.weight_block_n, weight_block_k=c.weight_block_k,
        name=f"{prefix}_shared_experts")
    return F.math.add(routed, shared, name=f"{prefix}_moe_output")


def build_dense_mlp(value, weights, config, *, prefix):
    """GLM dense SwiGLU block used by the first ``first_k_dense_replace`` layers."""

    gate_scale = weights.get("mlp.gate_proj.weight_scale_inv")
    up_scale = weights.get("mlp.up_proj.weight_scale_inv")
    if gate_scale is None and up_scale is None:
        gate_up = F.nn.dense_matmul_glu(
            value,
            weights["mlp.gate_proj.weight"],
            weights["mlp.up_proj.weight"],
            gate_clamp_max=config.swiglu_limit,
            up_clamp=config.swiglu_limit,
            name=f"{prefix}_dense_mlp_gate_up",
        )
    else:
        if gate_scale is None or up_scale is None:
            raise ValueError("GLM dense FP8 gate/up projections must provide both scale tensors.")
        gate_up = F.nn.matmul_glu(
            value, weights["mlp.gate_proj.weight"], weights["mlp.up_proj.weight"], gate_scale, up_scale,
            activation="silu", weight_block_n=config.weight_block_n, weight_block_k=config.weight_block_k,
            gate_clamp_max=config.swiglu_limit, up_clamp=config.swiglu_limit,
            name=f"{prefix}_dense_mlp_gate_up_fp8")
    return linear(gate_up, weights["mlp.down_proj.weight"],
                  weight_scale=weights.get("mlp.down_proj.weight_scale_inv"), config=config,
                  name=f"{prefix}_dense_mlp_down")
