# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""GLM-5.3-Flash synthetic checkpoints at reduced and published scale."""

import torch

from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.ir import DType

REDUCED = {
    "hidden_size": 16,
    "vocab_size": 32,
    "num_attention_heads": 2,
    "q_lora_rank": 8,
    "kv_lora_rank": 8,
    "qk_nope_head_dim": 8,
    "v_head_dim": 8,
    "moe_intermediate_size": 8,
    "intermediate_size": 8,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "layer_types": ("linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
                    "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention"),
    "mlp_layer_types": ("dense", "dense", "dense", "sparse", "sparse", "sparse", "sparse", "sparse"),
    "first_k_dense_replace": 3,
}

PUBLISHED = {
    "hidden_size": 4096,
    "vocab_size": 154880,
    "num_attention_heads": 64,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 256,
    "v_head_dim": 256,
    "moe_intermediate_size": 2048,
    "intermediate_size": 12288,
    "num_experts": 288,
    "num_experts_per_tok": 8,
    "layer_types": None,  # use the published 45-layer pattern
    "mlp_layer_types": None,
    "first_k_dense_replace": 3,
}


def _is_fp8_weight(key, shape, config=None, attention_kind=None):
    if len(shape) == 3:
        return key.endswith(("mlp.experts.gate_up_proj", "mlp.experts.down_proj"))
    if len(shape) != 2 or not key.endswith(".weight"):
        return False
    if key.startswith("self_attn.") or ".self_attn." in key:
        if attention_kind is not None and attention_kind != "deepseek_sparse_attention":
            return False
        if config is not None and ".layers." in key:
            try:
                layer = int(key.split(".layers.", 1)[1].split(".", 1)[0])
                layer_types = config["text_config"].get("layer_types", ())
                if layer_types and layer_types[layer] != "deepseek_sparse_attention":
                    return False
            except (ValueError, IndexError, KeyError):
                pass
        return key.endswith(("q_a_proj.weight", "q_b_proj.weight",
                             "kv_a_proj_with_mqa.weight", "o_proj.weight"))
    return (key.startswith("mlp.") or ".mlp." in key) and not key.endswith("gate.weight")


def _add_fp8_scales(config, shapes, attention_kind=None):
    quantization = config.get("quantization_config", {})
    block_n, block_k = quantization.get("weight_block_size", (128, 128))
    for key, shape in tuple(shapes.items()):
        if not _is_fp8_weight(key, shape, config, attention_kind):
            continue
        if len(shape) == 3:
            shapes[key + "_scale_inv"] = (
                shape[0], (shape[1] + block_n - 1) // block_n, (shape[2] + block_k - 1) // block_k)
        else:
            shapes[key + "_scale_inv"] = ((shape[0] + block_n - 1) // block_n,
                                           (shape[1] + block_k - 1) // block_k)


def configuration(scale=REDUCED):
    layer_types = scale["layer_types"]
    mlp_layer_types = scale["mlp_layer_types"]
    num_layers = len(layer_types) if layer_types else 45
    if layer_types is None:
        layer_types = tuple("deepseek_sparse_attention" if (layer + 1) % 4 == 0 else "linear_attention"
                            for layer in range(num_layers))
    if mlp_layer_types is None:
        mlp_layer_types = tuple("dense" if layer < 3 else "sparse" for layer in range(num_layers))
    kda_heads = scale.get("kda_num_heads", 64 if scale is PUBLISHED else 2)
    kda_dim = scale.get("kda_head_dim", 128 if scale is PUBLISHED else 8)
    kda_kernel = scale.get("kda_conv_kernel_size", 4)
    result = {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next_text",
        "tie_word_embeddings": False,
        "text_config": {
            "dtype": "bfloat16",
            "linear_attn_config": {"num_heads": kda_heads, "head_dim": kda_dim,
                                   "short_conv_kernel_size": kda_kernel,
                                   "gate_lower_bound": -5.0},
            "kda_num_heads": kda_heads,
            "kda_head_dim": kda_dim,
            "kda_conv_kernel_size": kda_kernel,
            "hidden_act": "silu",
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "mla_use_nope": True,
            "qk_rope_head_dim": 0,
            "head_dim": 0,
            "hidden_size": scale["hidden_size"],
            "vocab_size": scale["vocab_size"],
            "num_hidden_layers": num_layers,
            "num_attention_heads": scale["num_attention_heads"],
            "q_lora_rank": scale["q_lora_rank"],
            "kv_lora_rank": scale["kv_lora_rank"],
            "qk_nope_head_dim": scale["qk_nope_head_dim"],
            "v_head_dim": scale["v_head_dim"],
            "moe_intermediate_size": scale["moe_intermediate_size"],
            "intermediate_size": scale["intermediate_size"],
            "shared_expert_intermediate_size": scale["moe_intermediate_size"],
            "n_routed_experts": scale["num_experts"],
            "n_shared_experts": 1,
            "num_experts_per_tok": scale["num_experts_per_tok"],
            "routed_scaling_factor": 2.5,
            "swiglu_limit": 10.0,
            "rms_norm_eps": 1e-05,
            "first_k_dense_replace": scale["first_k_dense_replace"],
            "layer_types": list(layer_types),
            "mlp_layer_types": list(mlp_layer_types),
            "pad_token_id": 0,
            "mhc": bool(scale.get("mhc", False)),
            "hc_mult": int(scale.get("hc_mult", 4)),
            "hc_sinkhorn_iters": int(scale.get("hc_sinkhorn_iters", 2)),
            "hc_eps": float(scale.get("hc_eps", 1e-6)),
        },
    }
    if scale.get("indexer_types") is not None:
        result["text_config"].update({
            "indexer_types": list(scale["indexer_types"]),
            "index_topk": int(scale["index_topk"]),
            "index_head_dim": int(scale["index_head_dim"]),
            "index_n_heads": int(scale["index_n_heads"]),
            "index_kpool": int(scale["index_kpool"]),
            "index_kpool_always_select_tail": bool(scale.get("index_kpool_always_select_tail", True)),
        })
    if scale.get("quantized", False):
        result["quantization_config"] = {
            "quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "e4m3",
            "weight_block_size": list(scale.get("fp8_block_size", (128, 128))),
        }
    return result


def _hc_shapes(text):
    if not text.get("mhc", False):
        return {}
    hc = text["hc_mult"]
    coefficients = (2 + hc) * hc
    return {
        "hc_attn_fn": (coefficients, hc * text["hidden_size"]),
        "hc_attn_base": (coefficients,),
        "hc_attn_scale": (3,),
        "hc_ffn_fn": (coefficients, hc * text["hidden_size"]),
        "hc_ffn_base": (coefficients,),
        "hc_ffn_scale": (3,),
    }


def _weight_shapes(config, layer, mlp_kind="sparse"):
    text = config["text_config"]
    h, e, i = text["hidden_size"], text["n_routed_experts"], text["moe_intermediate_size"]
    heads, nope, v = text["num_attention_heads"], text["qk_nope_head_dim"], text["v_head_dim"]
    q_lora, kv_lora = text["q_lora_rank"], text["kv_lora_rank"]
    if text["layer_types"][layer] == "linear_attention":
        raise ValueError("KDA layer checkpoints are not materialized yet")
    weights = {
        "input_layernorm.weight": (h, ),
        "post_attention_layernorm.weight": (h, ),
        "self_attn.q_a_proj.weight": (q_lora, h),
        "self_attn.q_a_layernorm.weight": (q_lora, ),
        "self_attn.q_b_proj.weight": (heads * nope, q_lora),
        "self_attn.kv_a_proj_with_mqa.weight": (kv_lora, h),
        "self_attn.kv_a_layernorm.weight": (kv_lora, ),
        "self_attn.kv_b_proj.weight": (heads * (nope + v), kv_lora),
        "self_attn.o_proj.weight": (h, heads * nope),
    }
    weights.update(_hc_shapes(text))
    if text.get("indexer_types") and text["indexer_types"][layer] == "full":
        weights.update({
            "self_attn.indexer.wq_b.weight":
            (text["index_n_heads"] * text["index_head_dim"], q_lora),
            "self_attn.indexer.wk.weight": (text["index_head_dim"], h),
            "self_attn.indexer.k_norm.weight": (text["index_head_dim"],),
            "self_attn.indexer.k_norm.bias": (text["index_head_dim"],),
            "self_attn.indexer.weights_proj.weight": (text["index_n_heads"], h),
            "self_attn.indexer.index_kpool_compress_ape":
            (text["index_kpool"], text["index_head_dim"]),
            "self_attn.indexer.index_kpool_compress_gate": (text["index_head_dim"], h),
        })
    if mlp_kind == "dense":
        intermediate = text.get("intermediate_size", i)
        weights.update({
            "mlp.gate_proj.weight": (intermediate, h),
            "mlp.up_proj.weight": (intermediate, h),
            "mlp.down_proj.weight": (h, intermediate),
        })
    else:
        weights.update({
            "mlp.gate.weight": (e, h),
            "mlp.gate.e_score_correction_bias": (e, ),
            "mlp.experts.gate_up_proj": (e, 2 * i, h),
            "mlp.experts.down_proj": (e, h, i),
            "mlp.shared_experts.gate_proj.weight": (i, h),
            "mlp.shared_experts.up_proj.weight": (i, h),
            "mlp.shared_experts.down_proj.weight": (h, i),
        })
    if mlp_kind == "sparse" and text.get("expert_layout") == "separate":
        weights.pop("mlp.experts.gate_up_proj")
        weights.pop("mlp.experts.down_proj")
        for expert in range(e):
            weights.update({
                f"mlp.experts.{expert}.gate_proj.weight": (i, h),
                f"mlp.experts.{expert}.up_proj.weight": (i, h),
                f"mlp.experts.{expert}.down_proj.weight": (h, i),
            })
    if config["text_config"]["layer_types"][layer] != "deepseek_sparse_attention":
        raise ValueError("only sparse MLA layer checkpoints are exercised")
    return weights


def kda_weight_shapes(config, mlp_kind="sparse"):
    """GLM-5.3 KDA (linear-attention) layer weights, published modeling naming."""

    text = config["text_config"]
    h = text["hidden_size"]
    heads = text.get("kda_num_heads", 64)
    dim = text.get("kda_head_dim", 128)
    kernel = text.get("kda_conv_kernel_size", 4)
    qkv = heads * dim
    shapes = {
        "input_layernorm.weight": (h, ),
        "post_attention_layernorm.weight": (h, ),
        "self_attn.q_proj.weight": (qkv, h),
        "self_attn.k_proj.weight": (qkv, h),
        "self_attn.v_proj.weight": (qkv, h),
        "self_attn.q_conv1d.weight": (qkv, 1, kernel),
        "self_attn.k_conv1d.weight": (qkv, 1, kernel),
        "self_attn.v_conv1d.weight": (qkv, 1, kernel),
        "self_attn.f_a_proj.weight": (dim, h),
        "self_attn.f_b_proj.weight": (qkv, dim),
        "self_attn.dt_bias": (qkv, ),
        "self_attn.A_log": (heads, ),
        "self_attn.b_proj.weight": (heads, h),
        "self_attn.g_a_proj.weight": (dim, h),
        "self_attn.g_b_proj.weight": (qkv, dim),
        "self_attn.o_norm.weight": (dim, ),
        "self_attn.o_proj.weight": (h, qkv),
    }
    shapes.update(_hc_shapes(text))
    if mlp_kind == "dense":
        shapes.update({
            "mlp.gate_proj.weight": (text.get("intermediate_size", text["moe_intermediate_size"]), h),
            "mlp.up_proj.weight": (text.get("intermediate_size", text["moe_intermediate_size"]), h),
            "mlp.down_proj.weight": (h, text.get("intermediate_size", text["moe_intermediate_size"])),
        })
    else:
        shapes.update({
            "mlp.gate.weight": (text["n_routed_experts"], h),
            "mlp.gate.e_score_correction_bias": (text["n_routed_experts"], ),
            "mlp.experts.gate_up_proj": (text["n_routed_experts"], 2 * text["moe_intermediate_size"], h),
            "mlp.experts.down_proj": (text["n_routed_experts"], h, text["moe_intermediate_size"]),
            "mlp.shared_experts.gate_proj.weight": (text["moe_intermediate_size"], h),
            "mlp.shared_experts.up_proj.weight": (text["moe_intermediate_size"], h),
            "mlp.shared_experts.down_proj.weight": (h, text["moe_intermediate_size"]),
        })
        if text.get("expert_layout") == "separate":
            shapes.pop("mlp.experts.gate_up_proj")
            shapes.pop("mlp.experts.down_proj")
            for expert in range(text["n_routed_experts"]):
                shapes.update({
                    f"mlp.experts.{expert}.gate_proj.weight": (text["moe_intermediate_size"], h),
                    f"mlp.experts.{expert}.up_proj.weight": (text["moe_intermediate_size"], h),
                    f"mlp.experts.{expert}.down_proj.weight": (h, text["moe_intermediate_size"]),
                })
    return shapes


def checkpoint_layer(scale=REDUCED, *, requested=None):
    """Index of the sparse MLA layer the importer exercises for ``scale``.

    ``requested`` selects a specific sparse-MLA layer index (all sparse-MLA
    layers share identical shapes, so any index works).
    """

    text = configuration(scale)["text_config"]
    valid = (index for index, kind in enumerate(text["layer_types"])
             if kind == "deepseek_sparse_attention" and text["mlp_layer_types"][index] == "sparse")
    if requested is None:
        return next(valid)
    if requested not in set(valid):
        raise ValueError(f"layer {requested} is not a sparse MLA layer")
    return requested


def checkpoint(scale=REDUCED, *, seed=20260916, with_values=True, layer=None, kind="mla",
               full_model=False, expert_layout="packed"):
    config = configuration(scale)
    config["text_config"]["expert_layout"] = expert_layout
    text = config["text_config"]
    # keep kda dims visible to the importer via text_config keys
    text.setdefault("kda_num_heads", 64 if scale is PUBLISHED else 2)
    text.setdefault("kda_head_dim", 128 if scale is PUBLISHED else 8)
    text.setdefault("kda_conv_kernel_size", 4)
    shapes = {
        "model.language_model.embed_tokens.weight": (text["vocab_size"], text["hidden_size"]),
        "model.language_model.norm.weight": (text["hidden_size"], ),
        "lm_head.weight": (text["vocab_size"], text["hidden_size"]),
    }
    if full_model:
        for layer_index, attention_kind in enumerate(text["layer_types"]):
            mlp_kind = text["mlp_layer_types"][layer_index]
            if attention_kind == "linear_attention":
                layer_shapes = kda_weight_shapes(config, mlp_kind=mlp_kind)
            else:
                layer_shapes = _weight_shapes(config, layer_index, mlp_kind=mlp_kind)
            if scale.get("quantized", False):
                _add_fp8_scales(config, layer_shapes, attention_kind)
            for key, shape in layer_shapes.items():
                shapes[f"model.language_model.layers.{layer_index}.{key}"] = shape
    elif kind == "kda":
        if layer is not None and text["layer_types"][layer] == "linear_attention" \
                and text["mlp_layer_types"][layer] == "sparse":
            pass
        else:
            layer = next(index for index, kind_name in enumerate(text["layer_types"])
                         if kind_name == "linear_attention" and text["mlp_layer_types"][index] == "sparse")
        layer_shapes = kda_weight_shapes(config, mlp_kind=text["mlp_layer_types"][layer])
        if scale.get("quantized", False):
            _add_fp8_scales(config, layer_shapes, text["layer_types"][layer])
        for key, shape in layer_shapes.items():
            shapes[f"model.language_model.layers.{layer}.{key}"] = shape
    else:
        layer = checkpoint_layer(scale, requested=layer)
        for key, shape in _weight_shapes(config, layer).items():
            shapes[f"model.language_model.layers.{layer}.{key}"] = shape
    quantized = scale.get("quantized", False)
    infos = {
        key: TensorInfo(
            key,
            DType.FLOAT32 if key.endswith(("self_attn.dt_bias", "self_attn.A_log", "_scale_inv",
                                            "hc_attn_base", "hc_attn_scale",
                                            "hc_ffn_base", "hc_ffn_scale",
                                            "mlp.gate.e_score_correction_bias"))
            else DType.FLOAT8_E4M3FN if quantized and _is_fp8_weight(key, shape, config)
            else DType.BFLOAT16,
            shape, "model.safetensors")
        for key, shape in shapes.items()
    }

    values = {}
    if with_values:
        generator = torch.Generator().manual_seed(seed)
        values = {
            key: (torch.randn(shape, generator=generator) * 0.1).to(getattr(torch, infos[key].dtype.value))
            for key, shape in shapes.items()
        }
    return MemoryCheckpoint(config, infos, values)
