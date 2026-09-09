# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Editable decoder construction; no model-specific behavior in codegen."""

from triton.flagmega.ir import F, tensor_type


def linear(value, weight, *, name=None):
    return F.math.matmul(value, weight, transpose_b=True, name=name)


def rms_norm(value, weight, epsilon, *, name):
    # Qwen3.5 uses Gemma-style (1 + w) in FP32, without the older Qwen3
    # intermediate BF16 normalized-value/scale rounding boundary.
    wide_weight = F.tensors.cast(weight, "float32")
    one = F.builtin.splat_const(wide_weight.type, 1.0)
    zero = F.builtin.splat_const(wide_weight.type, 0.0)
    scale = F.math.add(wide_weight, one)
    stats = F.nn.norm_stats(value, axis=-1, use_mean=False)
    return F.nn.norm_apply(value, stats, scale, zero, axis=-1, epsilon=epsilon, use_mean=False,
                           round_before_scale=False, output_dtype="bfloat16", name=name)


def build_linear_attention(value, state, layer_id, weights, config, *, prefix):
    w = lambda name: weights[f"linear_attn.{name}"]
    layer_state = F.nn.gated_delta_net_state_slice(state, layer_id, name=f"{prefix}_state_view")
    qkv = linear(value, w("in_proj_qkv.weight"), name=f"{prefix}_qkv")
    z = linear(value, w("in_proj_z.weight"), name=f"{prefix}_z")
    convolution = F.nn.gated_delta_net_convolution(qkv, layer_state, w("conv1d.weight"),
                                                   conv_kernel_size=config.conv_kernel_size,
                                                   name=f"{prefix}_convolution")
    convolved, updated = F.tensors.get_items(convolution, 0, 1, name_prefix=f"{prefix}_conv")
    recurrent = F.nn.gated_delta_net_recurrent_core(updated, convolved, z, value, w("in_proj_b.weight"),
                                                    w("in_proj_a.weight"), w("A_log"), w("dt_bias"), w("norm.weight"),
                                                    num_key_heads=config.num_key_heads,
                                                    num_value_heads=config.num_value_heads,
                                                    key_head_dim=config.key_head_dim,
                                                    value_head_dim=config.value_head_dim, epsilon=config.epsilon,
                                                    name=f"{prefix}_recurrent")
    output = F.tensors.get_item(recurrent, 0)
    return linear(output, w("out_proj.weight"), name=f"{prefix}_attention_output")


def partial_rope(value, cosine, sine, config, *, name):
    return F.nn.rope(value, cosine, sine, rotary_dim=config.rotary_dim, name=name)


def build_full_attention(value, state, layer_id, advance, cosine, sine, weights, config, *, prefix):
    w = lambda name: weights[f"self_attn.{name}"]
    c = config
    tokens = value.type.shape[0].fixed_value
    projected = linear(value, w("q_proj.weight"), name=f"{prefix}_q_gate")
    per_head = F.tensors.reshape(projected, (tokens, c.num_attention_heads, 2 * c.head_dim))
    # Each head stores [query, gate]; splitting the flat projection in half
    # would exchange heads and gates while retaining apparently valid shapes.
    query = F.tensors.slice(per_head, starts=(0, ), ends=(c.head_dim, ), axes=(-1, ), name=f"{prefix}_query_slice")
    gate = F.tensors.slice(per_head, starts=(c.head_dim, ), ends=(None, ), axes=(-1, ), name=f"{prefix}_gate_slice")
    key = F.tensors.reshape(linear(value, w("k_proj.weight")), (tokens, c.num_key_value_heads, c.head_dim))
    val = F.tensors.reshape(linear(value, w("v_proj.weight")), (tokens, c.num_key_value_heads, c.head_dim))
    query = rms_norm(query, w("q_norm.weight"), c.epsilon, name=f"{prefix}_q_norm")
    key = rms_norm(key, w("k_norm.weight"), c.epsilon, name=f"{prefix}_k_norm")
    query = partial_rope(query, cosine, sine, c, name=f"{prefix}_q_rope")
    key = partial_rope(key, cosine, sine, c, name=f"{prefix}_k_rope")
    packed_q, packed_k, packed_v = (F.tensors.pack(item, 8, axis=-1) for item in (query, key, val))
    no_advance = F.builtin.scalar_const(tensor_type("bool", ()), False)
    layout = ("seq", "head", "dim")
    state = F.nn.update_paged_attention_kv_cache(packed_k, state, layer_id, no_advance, cache_kind="key", layout=layout)
    state = F.nn.update_paged_attention_kv_cache(packed_v, state, layer_id, advance, cache_kind="value", layout=layout)
    attended = F.nn.paged_attention(packed_q, state, layer_id, scale=c.head_dim**-0.5, layout=layout,
                                    hidden_size=c.query_size)
    scalar = F.tensors.unpack(attended, axis=-1)
    gated = F.math.mul(scalar, F.math.sigmoid(gate, name=f"{prefix}_attention_gate"))
    output = linear(F.tensors.reshape(gated, (tokens, c.query_size)), w("o_proj.weight"),
                    name=f"{prefix}_attention_output")
    return output, state


def build_moe(value, weights, config, *, prefix):
    c = config
    tokens = value.type.shape[0].fixed_value
    router_logits = linear(value, weights["mlp.gate.weight"], name=f"{prefix}_router_logits")
    probabilities = F.nn.softmax(F.tensors.cast(router_logits, "float32"), axis=-1, name=f"{prefix}_router_softmax")
    top = F.tensors.top_k(probabilities, k=c.num_experts_per_tok, name=f"{prefix}_router_top_k")
    scores, ids = F.tensors.get_items(top, 0, 1, name_prefix=f"{prefix}_router")
    total = F.math.reduce_sum(scores, axes=(-1, ), keep_dims=True)
    scores = F.math.div(scores, F.tensors.broadcast_to(total, shape=(tokens, c.num_experts_per_tok)),
                        name=f"{prefix}_router_weights")
    ones = F.builtin.splat_const(tensor_type("float32", (c.num_experts, 1)), 1.0)
    routed = F.nn.sparse_experts(value, ids, scores, ones, weights["mlp.experts.gate_proj"], ones, ones,
                                 weights["mlp.experts.down_proj"], ones, ones, weights["mlp.experts.up_proj"], ones,
                                 name=f"{prefix}_routed_experts")
    shared = F.nn.dense_matmul_glu(value, weights["mlp.shared_expert.gate_proj.weight"],
                                   weights["mlp.shared_expert.up_proj.weight"], activation="silu")
    shared = linear(shared, weights["mlp.shared_expert.down_proj.weight"])
    shared_gate = F.math.sigmoid(linear(value, weights["mlp.shared_expert_gate.weight"]), name=f"{prefix}_shared_gate")
    scaled = F.math.mul(shared, F.tensors.broadcast_to(shared_gate, shape=(tokens, c.hidden_size)),
                        name=f"{prefix}_shared_scaled")
    return F.math.add(routed, scaled, name=f"{prefix}_moe_output")
