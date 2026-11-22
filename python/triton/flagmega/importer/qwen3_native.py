# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Qwen3 native-byte decode graph with explicit K230 arithmetic contracts."""

from triton.flagmega.errors import ImporterError
from triton.flagmega.importer.source import attach_import_source_locations
from triton.flagmega.ir import DType, IRBuilder, PURE, TupleType, effect, get_definition, tensor_type, verify_module


NATIVE_PROFILE = "k230_q8_fp16_v1"
NORM_PROFILE = "ggml_f32_v1"
ATTENTION_PROFILE = "ggml_cpu_f16_1t_v1"
ROWWISE_ATTENTION_PROFILE = "ggml_cpu_f16_rowwise_1t_v1"
VERIFY_PHASE = "verify"
VERIFY_TOKENS = range(2, 9)


def validate_native_profile(profile, execution_phase):
    if profile not in (None, NATIVE_PROFILE):
        raise ImporterError(f"Qwen3 native_compute_profile must be None or {NATIVE_PROFILE!r}.")
    if profile is not None and execution_phase not in {"decode-1", VERIFY_PHASE}:
        raise ImporterError("Qwen3 native arithmetic supports decode-1 only, or an explicit verify chunk.")
    if profile is None and execution_phase == VERIFY_PHASE:
        raise ImporterError("Qwen3 verify requires the explicit native compute profile.")
    return profile


def import_native_qwen3(importer, *, full_model):
    """Import decode-1, or a fixed verify chunk whose row i runs at position seq_lens[0] + i.

    Every verify row keeps the decode-1 arithmetic. Attention is causal inside the chunk; the
    last layer advances seq_lens by the chunk length and the caller commits accepted rows.
    """
    config = importer.config
    tokens = importer.num_tokens
    verify = importer.execution_phase == VERIFY_PHASE
    if verify and (not full_model or tokens not in VERIFY_TOKENS):
        raise ImporterError("Qwen3 native verify imports the full model with two to eight token rows.")
    prefixes = importer.layer_prefixes if full_model else (importer.layer_prefix,)
    hidden_type = tensor_type(DType.FLOAT32, (tokens, config.hidden_size))
    state_type = importer.state_config.ref_type
    metadata = {
        "architecture": "Qwen3ForCausalLM", "model_type": "qwen3", "mode": VERIFY_PHASE if verify else "decode-1",
        "tokens_per_call": tokens, "num_layers": len(prefixes), "num_hidden_layers": config.num_hidden_layers,
        "layer": None if full_model else importer.layer, "layer_prefixes": prefixes,
        "model_prefix": importer.model_prefix, "revision": importer.revision,
        "output_boundary": "logits_fp32_and_greedy_token" if full_model else "final_norm_hidden_fp32",
        "vocab_size": config.vocab_size, "padding_idx": config.padding_idx,
        "hidden_size": config.hidden_size, "intermediate_size": config.intermediate_size,
        "num_attention_heads": config.num_attention_heads, "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim, "rope_theta": config.rope_theta,
        "max_position_embeddings": config.max_position_embeddings, "tie_word_embeddings": config.tie_word_embeddings,
        "native_compute_profile": NATIVE_PROFILE, "norm_compute_profile": NORM_PROFILE,
        "attention_compute_profile": ROWWISE_ATTENTION_PROFILE if verify else ATTENTION_PROFILE,
        "paged_attention": {
            "block_size": importer.block_size, "num_blocks": importer.num_blocks,
            "cache_layout": ["NumBlocks", "NumLayers", "KV", "BlockSize", "NumKVHeads", "HeadDim"],
            "vectorized_axes": ["HeadDim"], "lanes": [8], "cache_dtype": "float16",
        },
    }
    if verify:
        metadata.update(k230_static_program_profile=NATIVE_PROFILE, verify={
            "schema": "flagmega.qwen3-verify/v1", "rows": tokens,
            "row_positions": "row i uses logical position seq_lens[0] + i; seq_lens is read on entry",
            "attention": "row i reads positions 0..seq_lens[0]+i with its decode-1 padded length",
            "exit_state": "query_start_loc=[0,rows], slot_mapping=[entry seq_lens], seq_lens=entry+rows",
            "commit": "caller keeps r rows, restores rejected KV rows, and sets decode-1 metadata for r",
        })
    builder = IRBuilder(dialect="high_level", stage="imported", metadata=metadata)

    def call(op, inputs, node_id, attrs=None, node_effect=PURE):
        definition = get_definition(op)
        normalized = definition.normalize_attrs(attrs or {})
        return builder.call(op, inputs, definition.infer_type(inputs, normalized), id=node_id,
                            attrs=definition.ir_attrs(normalized), effect=node_effect)

    def item(value, index, name):
        return call("builtin.get_item", [value], name, {"index": index})

    def scalar(name, value, dtype):
        return call("builtin.scalar_const", [], name, {"result_type": tensor_type(dtype, ()), "value": value})

    def info_at(key, shape, encoding):
        info = importer.checkpoint.tensor_info(key)
        if info.shape != shape or info.quant_type != encoding or (encoding is None and info.dtype != DType.FLOAT32):
            expected = encoding or "float32"
            raise ImporterError(f"Native Qwen3 tensor {key!r} requires {expected} shape {shape}, got {info.quant_type or info.dtype.value} {info.shape}.")
        return info

    def weight(name, key, shape, encoding, group=None):
        info = info_at(key, shape, encoding)
        return builder.weight(name, info.storage_type, source=info.source, key=key,
                              id="w_" + name.replace(".", "_"),
                              metadata={} if group is None else {"rdata_group": group})

    def norm(value, scale, name):
        return call("nn.rms_norm", [value, scale], name,
                    {"epsilon": config.epsilon, "weight_bias": 0.0, "compute_profile": NORM_PROFILE})

    def linear(value, matrix, scale, name, encoding="IQ4_XS"):
        return call("math.block_scaled_matmul", [value, matrix, scale], name,
                    {"weight_block_n": 128, "weight_block_k": 128, "weight_encoding": encoding,
                     "output_data_type": "float32", "native_compute_profile": NATIVE_PROFILE})

    def scale_spec(key, rows, columns):
        return key.removesuffix("weight") + "weight_scale_inv", ((rows + 127) // 128, (columns + 127) // 128), None

    input_ids = builder.var("input_ids", tensor_type(DType.INT32, (tokens,)), id="input_ids")
    entry_state = builder.var("paged_attention_kv_cache", state_type, id="paged_attention_kv_cache")
    embedding = weight("embed_tokens.weight", importer.model_prefix + "embed_tokens.weight",
                       (config.vocab_size, config.hidden_size), "IQ4_XS")
    hidden = call("nn.embedding", [input_ids, embedding], "token_embedding",
                  {"padding_idx": config.padding_idx, "weight_encoding": "IQ4_XS", "output_data_type": "float32"})
    rotary = call("nn.rotary_embedding", [hidden, entry_state], "main_rotary_embedding",
                  {"head_dim": config.head_dim, "theta": config.rope_theta, "math_mode": "ggml"},
                  effect("read", "paged_attention_kv_cache"))
    position = (item(rotary, 0, "main_rotary_cos"), item(rotary, 1, "main_rotary_sin"))

    specs = {
        "input_norm_weight": ("input_layernorm.weight", (config.hidden_size,), None),
        "q_weight": ("self_attn.q_proj.weight", (config.query_size, config.hidden_size), "IQ4_XS"),
        "k_weight": ("self_attn.k_proj.weight", (config.kv_size, config.hidden_size), "IQ4_XS"),
        "v_weight": ("self_attn.v_proj.weight", (config.kv_size, config.hidden_size), "IQ4_XS"),
        "q_norm_weight": ("self_attn.q_norm.weight", (config.head_dim,), None),
        "k_norm_weight": ("self_attn.k_norm.weight", (config.head_dim,), None),
        "o_weight": ("self_attn.o_proj.weight", (config.hidden_size, config.query_size), "IQ4_XS"),
        "o_scale": scale_spec("self_attn.o_proj.weight", config.hidden_size, config.query_size),
        "post_norm_weight": ("post_attention_layernorm.weight", (config.hidden_size,), None),
        "gate_weight": ("mlp.gate_proj.weight", (config.intermediate_size, config.hidden_size), "IQ4_XS"),
        "up_weight": ("mlp.up_proj.weight", (config.intermediate_size, config.hidden_size), "IQ4_XS"),
        "gate_scale": scale_spec("mlp.gate_proj.weight", config.intermediate_size, config.hidden_size),
        "up_scale": scale_spec("mlp.up_proj.weight", config.intermediate_size, config.hidden_size),
        "down_weight": ("mlp.down_proj.weight", (config.hidden_size, config.intermediate_size), "IQ4_XS"),
        "down_scale": scale_spec("mlp.down_proj.weight", config.hidden_size, config.intermediate_size),
    }

    def parameter(name, value_type):
        return builder.var(name, value_type, id="decode_layer_" + name,
                           metadata={"function_parameter": "decode_layer"})

    decode_hidden = parameter("hidden", hidden_type)
    decode_state = parameter("state", state_type)
    params = {name: parameter(name, info_at(prefixes[0] + key, shape, encoding).storage_type)
              for name, (key, shape, encoding) in specs.items()}
    layer_id = parameter("layer_id", tensor_type(DType.INT32, ()))
    advance = parameter("advance_sequence", tensor_type(DType.BOOL, ()))
    cosine, sine = (parameter(name, value.type) for name, value in zip(("rotary_cos", "rotary_sin"), position))
    normalized = norm(decode_hidden, params["input_norm_weight"], "decode_layer_input_norm")
    none = call("builtin.none", [], "decode_layer_none")
    qkv = call("nn.qkv_parallel_linear", [normalized, params["q_weight"], params["k_weight"], params["v_weight"], *([none] * 9)],
               "decode_layer_qkv_projection",
               {"num_heads": config.num_attention_heads, "num_kv_heads": config.num_key_value_heads,
                "output_data_type": "float32", "q_weight_encoding": "IQ4_XS", "k_weight_encoding": "IQ4_XS",
                "v_weight_encoding": "IQ4_XS", "native_compute_profile": NATIVE_PROFILE})
    projected = []
    for index, (name, heads) in enumerate((("query", config.num_attention_heads), ("key", config.num_key_value_heads), ("value", config.num_key_value_heads))):
        flat = item(qkv, index, "decode_layer_" + name + "_flat")
        projected.append(call("tensors.reshape", [flat], "decode_layer_" + name, {"shape": (tokens, heads, config.head_dim)}))
    q_stats = call("nn.norm_stats", [projected[0]], "decode_layer_query_stats",
                   {"axis": -1, "use_mean": False, "compute_profile": NORM_PROFILE})
    k_stats = call("nn.norm_stats", [projected[1]], "decode_layer_key_stats",
                   {"axis": -1, "use_mean": False, "compute_profile": NORM_PROFILE})
    shaped_qkv = call("builtin.tuple", projected, "decode_layer_qkv_shaped")
    zero_bias = call("builtin.splat_const", [], "decode_layer_zero_bias",
                     {"result_type": tensor_type(DType.FLOAT32, (config.head_dim,)), "value": 0.0})
    layout = ("seq", "head", "dim")
    qkv_state = call("nn.qkv_rope_with_cache",
                     [shaped_qkv, params["q_norm_weight"], params["k_norm_weight"], zero_bias, zero_bias,
                      cosine, sine, decode_state, layer_id, advance, q_stats, k_stats],
                     "decode_layer_qkv_rope_cache",
                     {"q_axis": -1, "k_axis": -1, "q_epsilon": config.epsilon, "k_epsilon": config.epsilon,
                      "q_use_mean": False, "k_use_mean": False, "qkv_layout": layout, "attention_layout": layout,
                      "norm_compute_profile": NORM_PROFILE, "query_output_lanes": ()},
                     effect("read_write", "paged_attention_kv_cache"))
    query = item(qkv_state, 0, "decode_layer_query_rotated")
    updated_state = item(qkv_state, 1, "decode_layer_updated_state")
    attended = call("nn.paged_attention", [query, updated_state, layer_id], "decode_layer_paged_attention",
                    {"scale": config.head_dim ** -0.5, "layout": layout, "hidden_size": config.query_size,
                     "compute_profile": metadata["attention_compute_profile"]}, effect("read", "paged_attention_kv_cache"))
    merged = call("tensors.reshape", [attended], "decode_layer_attention_merged", {"shape": (tokens, config.query_size)})
    attention_output = linear(merged, params["o_weight"], params["o_scale"], "decode_layer_attention_output")
    residual = call("math.add", [decode_hidden, attention_output], "decode_layer_after_attention")
    mlp_input = norm(residual, params["post_norm_weight"], "decode_layer_post_attention_norm")
    glu = call("nn.matmul_glu", [mlp_input, params["gate_weight"], params["up_weight"], params["gate_scale"], params["up_scale"]],
               "decode_layer_mlp_gate_up", {"activation": "silu", "weight_block_n": 128, "weight_block_k": 128,
                                           "gate_weight_encoding": "IQ4_XS", "up_weight_encoding": "IQ4_XS",
                                           "native_compute_profile": NATIVE_PROFILE})
    mlp_output = linear(glu, params["down_weight"], params["down_scale"], "decode_layer_mlp_down")
    decode_output = call("math.add", [residual, mlp_output], "decode_layer_output")
    decode_parameters = (decode_hidden, decode_state, *params.values(), layer_id, advance, cosine, sine)

    state = entry_state
    for index, prefix in enumerate(prefixes):
        values = [weight(f"layers.{index}.{key}", prefix + key, shape, encoding,
                         {"name": "qwen3.layer." + key, "index": index, "count": len(prefixes)})
                  for key, shape, encoding in specs.values()]
        actual_layer = index if full_model else importer.layer
        layer_index = scalar(f"layer_{index}_id", actual_layer, DType.INT32)
        advance_value = scalar(f"layer_{index}_advance_sequence", index + 1 == len(prefixes), DType.BOOL)
        invocation = builder.call("builtin.call", [hidden, state, *values, layer_index, advance_value, *position],
                                  TupleType((hidden_type, state_type)), id=f"layer_{index}_decode_layer_call",
                                  attrs={"callee": "decode_layer"}, effect=effect("read_write", "paged_attention_kv_cache"),
                                  metadata={"layer_index": actual_layer})
        hidden = item(invocation, 0, f"layer_{index}_decoder_output")
        state = item(invocation, 1, "updated_state" if index + 1 == len(prefixes) else f"layer_{index}_updated_state")
    final_scale = weight("norm.weight", importer.model_prefix + "norm.weight", (config.hidden_size,), None)
    final_hidden = norm(hidden, final_scale, "final_norm")
    if full_model:
        head_key = importer.model_prefix + "embed_tokens.weight" if config.tie_word_embeddings else "lm_head.weight"
        head_encoding = importer.checkpoint.tensor_info(head_key).quant_type
        if head_encoding not in ("IQ4_XS", "Q6_K"):
            raise ImporterError("Native Qwen3 output head requires original IQ4_XS or Q6_K storage.")
        head = embedding if config.tie_word_embeddings else weight("lm_head.weight", head_key, (config.vocab_size, config.hidden_size), head_encoding)
        scale_key, scale_shape, _ = scale_spec(head_key, config.vocab_size, config.hidden_size)
        head_scale = weight("lm_head.weight_scale_inv", scale_key, scale_shape, None)
        logits = linear(final_hidden, head, head_scale, "logits", head_encoding)
        next_token = call("nn.greedy_sample", [logits], "next_token")
        builder.function("main", [input_ids, entry_state], [logits, next_token, state])
    else:
        builder.function("main", [input_ids, entry_state], [final_hidden, state])
    builder.function("decode_layer", decode_parameters, [decode_output, updated_state],
                     attrs={"calling_convention": "device", "noinline": True, "reusable": True})
    return verify_module(attach_import_source_locations(builder.build(entry="main"),
                                                       architecture="Qwen3ForCausalLM", revision=importer.revision))
