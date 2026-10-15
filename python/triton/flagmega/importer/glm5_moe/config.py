# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Validated contracts for the supported GLM-5.3-Flash decoder subset.

The parser uses the published configuration and weight names. Native FP8
block scales are represented explicitly in the shape/dtype contract; DSA
indexing remains a separate attention-state capability. This file is the
shape/dtype contract only; op construction lives in ``decoder``.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from triton.flagmega.errors import ImporterError
from triton.flagmega.ir import tensor_type


@dataclass(frozen=True)
class Glm5MoeConfig:
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    v_head_dim: int
    intermediate_size: int
    moe_intermediate_size: int
    shared_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    routed_scaling_factor: float
    first_k_dense_replace: int
    layer_types: tuple[str, ...]
    mlp_layer_types: tuple[str, ...]
    epsilon: float
    swiglu_limit: float
    kda_num_heads: int
    kda_head_dim: int
    kda_conv_kernel_size: int
    kda_gate_lower_bound: float | None
    tie_word_embeddings: bool
    padding_idx: int | None
    mhc: bool
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    index_topk: int
    index_head_dim: int
    index_n_heads: int
    index_kpool: int
    index_kpool_always_select_tail: bool
    indexer_types: tuple[str, ...]
    quantized: bool
    weight_block_n: int
    weight_block_k: int

    @property
    def query_size(self):
        return self.num_attention_heads * self.qk_nope_head_dim

    @property
    def kv_head_size(self):
        return self.qk_nope_head_dim + self.v_head_dim

    @classmethod
    def parse(cls, source, *, allow_mhc=False):
        text = source.get("text_config", source)
        quantization = source.get("quantization_config", {})
        if quantization is None:
            quantization = {}
        if not isinstance(quantization, Mapping):
            raise ImporterError("GLM-5.3 Flash quantization_config must be an object when present.")
        quant_method = quantization.get("quant_method")
        if quant_method is not None and str(quant_method).lower() != "fp8":
            raise ImporterError(
                "GLM-5.3 Flash supports only dynamic FP8 E4M3 quantization; "
                f"got quant_method={quant_method!r}.")
        quantized = str(quant_method).lower() == "fp8"
        if quantized:
            activation_scheme = str(quantization.get("activation_scheme", "")).lower()
            fmt = str(quantization.get("fmt", "")).lower()
            if activation_scheme != "dynamic" or fmt not in {"e4m3", "float8_e4m3fn"}:
                raise ImporterError(
                    "GLM-5.3 Flash FP8 requires activation_scheme='dynamic' and fmt='e4m3'.")
        block_size = quantization.get("weight_block_size", (128, 128))
        if (not isinstance(block_size, (list, tuple)) or len(block_size) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in block_size)):
            raise ImporterError("GLM-5.3 Flash FP8 weight_block_size must contain two positive integers.")
        if not isinstance(text, Mapping):
            raise ImporterError("GLM-5.3 Flash text_config must be an object.")
        if text.get("hidden_act", "silu") != "silu":
            raise ImporterError("GLM-5.3 Flash importer requires SiLU activations.")
        if text.get("scoring_func", "sigmoid") != "sigmoid":
            raise ImporterError("GLM-5.3 Flash importer requires sigmoid MoE scoring.")
        if text.get("topk_method", "noaux_tc") != "noaux_tc":
            raise ImporterError("GLM-5.3 Flash importer requires noaux_tc routing.")
        if text.get("mla_use_nope", False) is not True:
            raise ImporterError("GLM-5.3 Flash importer requires nope-only MLA.")
        mhc = text.get("mhc", False)
        if not isinstance(mhc, bool):
            raise ImporterError("GLM-5.3 Flash mhc must be a boolean when present.")
        if mhc and not allow_mhc:
            raise ImporterError(
                "GLM-5.3 Flash hyper-connections (mhc) are not supported by the single-block importer."
            )
        if mhc:
            hc_mult = text.get("hc_mult")
            hc_sinkhorn_iters = text.get("hc_sinkhorn_iters")
            hc_eps = text.get("hc_eps")
            if (isinstance(hc_mult, bool) or not isinstance(hc_mult, int) or hc_mult <= 0
                    or isinstance(hc_sinkhorn_iters, bool) or not isinstance(hc_sinkhorn_iters, int)
                    or hc_sinkhorn_iters <= 0):
                raise ImporterError("GLM-5.3 Flash hc_mult/hc_sinkhorn_iters must be positive integers.")
            try:
                hc_eps = float(hc_eps)
            except (TypeError, ValueError) as error:
                raise ImporterError("GLM-5.3 Flash hc_eps must be a positive finite number.") from error
            if not isfinite(hc_eps) or hc_eps <= 0:
                raise ImporterError("GLM-5.3 Flash hc_eps must be a positive finite number.")
        else:
            hc_mult, hc_sinkhorn_iters, hc_eps = 1, 1, 1e-6
        raw_indexer_types = text.get("indexer_types")
        if raw_indexer_types is None:
            indexer_types = ()
            index_topk = index_head_dim = index_n_heads = index_kpool = 0
            index_kpool_always_select_tail = True
        else:
            if (not isinstance(raw_indexer_types, (list, tuple))
                    or len(raw_indexer_types) != text.get("num_hidden_layers")
                    or any(kind not in {"full", "shared"} for kind in raw_indexer_types)):
                raise ImporterError(
                    "GLM-5.3 Flash indexer_types must classify every layer as full or shared.")
            indexer_types = tuple(raw_indexer_types)
            if indexer_types[0] != "full":
                raise ImporterError("GLM-5.3 Flash sparse-indexer schedule must begin with a full layer.")
            index_fields = {
                "index_topk": text.get("index_topk"),
                "index_head_dim": text.get("index_head_dim"),
                "index_n_heads": text.get("index_n_heads"),
                "index_kpool": text.get("index_kpool"),
            }
            if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                   for value in index_fields.values()):
                raise ImporterError("GLM-5.3 Flash sparse-indexer dimensions must be positive integers.")
            index_topk = index_fields["index_topk"]
            index_head_dim = index_fields["index_head_dim"]
            index_n_heads = index_fields["index_n_heads"]
            index_kpool = index_fields["index_kpool"]
            if index_topk % index_kpool:
                raise ImporterError("GLM-5.3 Flash sparse-indexer index_topk must be divisible by index_kpool.")
            index_kpool_always_select_tail = text.get("index_kpool_always_select_tail", True)
            if not isinstance(index_kpool_always_select_tail, bool):
                raise ImporterError("GLM-5.3 Flash index_kpool_always_select_tail must be boolean.")
        try:
            rope_head_dim = int(text.get("qk_rope_head_dim", 0) or 0)
            attention_head_dim = int(text.get("head_dim", 0) or 0)
        except (TypeError, ValueError) as error:
            raise ImporterError("GLM-5.3 Flash rope/head dimensions must be integers.") from error
        if rope_head_dim != 0 or attention_head_dim != 0:
            raise ImporterError("GLM-5.3 Flash importer requires the nope MLA layout (no rope head dim).")
        mapping = {
            "hidden_size": "hidden_size",
            "vocab_size": "vocab_size",
            "num_hidden_layers": "num_hidden_layers",
            "num_attention_heads": "num_attention_heads",
            "q_lora_rank": "q_lora_rank",
            "kv_lora_rank": "kv_lora_rank",
            "qk_nope_head_dim": "qk_nope_head_dim",
            "v_head_dim": "v_head_dim",
            "intermediate_size": "intermediate_size",
            "moe_intermediate_size": "moe_intermediate_size",
            "num_experts": "n_routed_experts",
            "num_experts_per_tok": "num_experts_per_tok",
            "n_shared_experts": "n_shared_experts",
            "first_k_dense_replace": "first_k_dense_replace",
        }
        values = {}
        nonnegative = {"first_k_dense_replace"}
        for name, key in mapping.items():
            value = text.get(key)
            if name == "intermediate_size" and value is None:
                # Older synthetic fixtures predate the dense-MLP field. Their
                # reduced dense layers use the same width as the shared MLP.
                value = text.get("shared_expert_intermediate_size",
                                 text.get("moe_intermediate_size"))
            minimum = 0 if name in nonnegative else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                qualifier = "non-negative" if minimum == 0 else "positive"
                raise ImporterError(f"GLM-5.3 Flash {key} must be a {qualifier} integer.")
            values[name] = value
        shared = text.get("shared_expert_intermediate_size",
                          values["moe_intermediate_size"] * values["n_shared_experts"])
        if isinstance(shared, bool) or not isinstance(shared, int) or shared <= 0:
            raise ImporterError("GLM-5.3 Flash shared_expert_intermediate_size must be a positive integer.")
        values["shared_intermediate_size"] = shared
        try:
            scaling = float(text.get("routed_scaling_factor", 1.0))
            epsilon = float(text.get("rms_norm_eps", 1e-5))
            limit = float(text.get("swiglu_limit", 0.0))
        except (TypeError, ValueError) as error:
            raise ImporterError("Invalid GLM-5.3 Flash numeric attributes.") from error
        if (not isfinite(scaling) or not isfinite(epsilon) or not isfinite(limit)
                or scaling <= 0 or epsilon <= 0 or limit <= 0):
            raise ImporterError("GLM-5.3 Flash scaling/epsilon/swiglu_limit must be positive.")
        linear = text.get("linear_attn_config")
        if not isinstance(linear, Mapping):
            raise ImporterError("GLM-5.3 Flash importer requires linear_attn_config for KDA layers.")
        for name, key in (("kda_num_heads", "num_heads"), ("kda_head_dim", "head_dim"),
                          ("kda_conv_kernel_size", "short_conv_kernel_size")):
            value = linear.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ImporterError(f"GLM-5.3 Flash linear_attn_config.{key} must be a positive integer.")
        raw_lower_bounds = {
            "linear_attn_config.gate_lower_bound": linear.get("gate_lower_bound"),
            "gate_lower_bound": text.get("gate_lower_bound"),
        }
        parsed_lower_bounds = {}
        for name, value in raw_lower_bounds.items():
            if value is None:
                continue
            try:
                parsed_lower_bounds[name] = float(value)
            except (TypeError, ValueError) as error:
                raise ImporterError(f"Invalid GLM-5.3 Flash {name}.") from error
        if len(parsed_lower_bounds) == 2:
            nested_value = parsed_lower_bounds["linear_attn_config.gate_lower_bound"]
            top_level_value = parsed_lower_bounds["gate_lower_bound"]
            if nested_value != top_level_value:
                raise ImporterError("GLM-5.3 Flash gate_lower_bound is duplicated with conflicting values.")
        if parsed_lower_bounds:
            lower_bound = next(iter(parsed_lower_bounds.values()))
            if not isfinite(lower_bound) or lower_bound >= 0:
                raise ImporterError("GLM-5.3 Flash gate_lower_bound must be a finite negative number.")
        else:
            raise ImporterError("GLM-5.3 Flash gate_lower_bound is required for KDA layers.")
        if values["qk_nope_head_dim"] != values["v_head_dim"]:
            raise ImporterError(
                "GLM-5.3 Flash importer requires equal MLA key and value head dimensions "
                "for the shared paged-attention cache."
            )
        layer_types = text.get("layer_types")
        mlp_layer_types = text.get("mlp_layer_types")
        known_attention = {"linear_attention", "deepseek_sparse_attention"}
        known_mlp = {"dense", "sparse"}
        for name, kinds, allowed in (("layer_types", layer_types, known_attention),
                                     ("mlp_layer_types", mlp_layer_types, known_mlp)):
            if (not isinstance(kinds, (list, tuple)) or len(kinds) != values["num_hidden_layers"]
                    or any(kind not in allowed for kind in kinds)):
                raise ImporterError(f"GLM-5.3 Flash {name} must classify every decoder layer.")
        expected_kda = tuple(index for index, kind in enumerate(layer_types) if kind == "linear_attention")
        expected_full = tuple(index for index, kind in enumerate(layer_types)
                              if kind == "deepseek_sparse_attention")
        for name, declared, expected in (("kda_layers", linear.get("kda_layers"), expected_kda),
                                         ("full_attn_layers", linear.get("full_attn_layers"), expected_full)):
            if declared is None:
                continue
            if (not isinstance(declared, (list, tuple))
                    or any(isinstance(index, bool) or not isinstance(index, int) for index in declared)):
                raise ImporterError(f"GLM-5.3 Flash linear_attn_config.{name} must be integer indices.")
            if tuple(declared) != expected:
                raise ImporterError(f"GLM-5.3 Flash linear_attn_config.{name} disagrees with layer_types.")
        if values["num_experts_per_tok"] > values["num_experts"]:
            raise ImporterError("GLM-5.3 Flash selected experts cannot exceed the expert bank.")
        tied = text.get("tie_word_embeddings", source.get("tie_word_embeddings", False))
        padding = text.get("pad_token_id", source.get("pad_token_id"))
        if not isinstance(tied, bool) or (padding is not None and
                                          (isinstance(padding, bool) or not isinstance(padding, int)
                                           or not 0 <= padding < values["vocab_size"])):
            raise ImporterError("GLM-5.3 Flash embedding configuration is invalid.")
        return cls(
            **values,
            routed_scaling_factor=scaling,
            layer_types=tuple(layer_types),
            mlp_layer_types=tuple(mlp_layer_types),
            epsilon=epsilon,
            swiglu_limit=limit,
            kda_num_heads=linear["num_heads"],
            kda_head_dim=linear["head_dim"],
            kda_conv_kernel_size=linear["short_conv_kernel_size"],
            kda_gate_lower_bound=lower_bound,
            tie_word_embeddings=tied,
            padding_idx=padding,
            mhc=mhc,
            hc_mult=hc_mult,
            hc_sinkhorn_iters=hc_sinkhorn_iters,
            hc_eps=hc_eps,
            index_topk=index_topk,
            index_head_dim=index_head_dim,
            index_n_heads=index_n_heads,
            index_kpool=index_kpool,
            index_kpool_always_select_tail=index_kpool_always_select_tail,
            indexer_types=indexer_types,
            quantized=quantized,
            weight_block_n=block_size[0],
            weight_block_k=block_size[1],
        )

    def state_kinds(self, layers):
        """Paged-attention and GDN layer counts exercised by ``layers``."""

        attention = sum(1 for layer in layers if self.layer_types[layer] == "deepseek_sparse_attention")
        linear = sum(1 for layer in layers if self.layer_types[layer] == "linear_attention")
        return attention, linear

    def weight_types(self, kind, mlp_kind="sparse", indexer_type=None):
        """Checkpoint keys for one decoder of ``kind`` with BF16 storage.

        Native FP8 weights require block-scale tensors and scaled matmul
        operators; they are not silently treated as BF16 by this importer.
        """

        h = self.hidden_size
        if mlp_kind not in ("dense", "sparse"):
            raise ImporterError(f"Unknown GLM-5.3 Flash MLP kind {mlp_kind!r}.")
        shapes = {
            "input_layernorm.weight": (h, ),
            "post_attention_layernorm.weight": (h, ),
        }
        if self.mhc:
            hc = self.hc_mult
            coefficients = (2 + hc) * hc
            shapes.update({
                "hc_attn_fn": (coefficients, hc * h),
                "hc_attn_base": (coefficients, ),
                "hc_attn_scale": (3, ),
                "hc_ffn_fn": (coefficients, hc * h),
                "hc_ffn_base": (coefficients, ),
                "hc_ffn_scale": (3, ),
            })
        if mlp_kind == "dense":
            shapes.update({
                "mlp.gate_proj.weight": (self.intermediate_size, h),
                "mlp.up_proj.weight": (self.intermediate_size, h),
                "mlp.down_proj.weight": (h, self.intermediate_size),
            })
        else:
            shapes.update({
                "mlp.gate.weight": (self.num_experts, h),
                "mlp.gate.e_score_correction_bias": (self.num_experts, ),
                "mlp.experts.gate_up_proj": (self.num_experts, 2 * self.moe_intermediate_size, h),
                "mlp.experts.down_proj": (self.num_experts, h, self.moe_intermediate_size),
            })
            if self.n_shared_experts:
                shapes.update({
                    "mlp.shared_experts.gate_proj.weight": (self.shared_intermediate_size, h),
                    "mlp.shared_experts.up_proj.weight": (self.shared_intermediate_size, h),
                    "mlp.shared_experts.down_proj.weight": (h, self.shared_intermediate_size),
                })
        if kind == "mla_attention":
            shapes.update({
                "self_attn.q_a_proj.weight": (self.q_lora_rank, h),
                "self_attn.q_a_layernorm.weight": (self.q_lora_rank, ),
                "self_attn.q_b_proj.weight": (self.query_size, self.q_lora_rank),
                "self_attn.kv_a_proj_with_mqa.weight": (self.kv_lora_rank, h),
                "self_attn.kv_a_layernorm.weight": (self.kv_lora_rank, ),
                "self_attn.kv_b_proj.weight": (self.num_attention_heads * self.kv_head_size, self.kv_lora_rank),
                "self_attn.o_proj.weight": (h, self.query_size),
            })
            if self.indexer_types and indexer_type != "shared":
                shapes.update({
                    "self_attn.indexer.wq_b.weight": (self.index_n_heads * self.index_head_dim,
                                                        self.q_lora_rank),
                    "self_attn.indexer.wk.weight": (self.index_head_dim, h),
                    "self_attn.indexer.k_norm.weight": (self.index_head_dim,),
                    "self_attn.indexer.k_norm.bias": (self.index_head_dim,),
                    "self_attn.indexer.weights_proj.weight": (self.index_n_heads, h),
                    "self_attn.indexer.index_kpool_compress_ape":
                    (self.index_kpool, self.index_head_dim),
                    "self_attn.indexer.index_kpool_compress_gate": (self.index_head_dim, h),
                })
        elif kind == "linear_attention":
            # GLM-5.3 KDA (Kimi Delta Attention) per the published modeling:
            # separate q/k/v projections, a depthwise short convolution, a
            # low-rank per-channel forget gate (f_a/f_b with per-head A_log and
            # per-channel dt_bias), a per-head write-strength projection, and
            # a gated RMSNorm before the out projection. dt_bias/A_log stay in
            # FP32 (`_keep_in_fp32_modules_strict` in the reference modeling).
            qkv = self.kda_num_heads * self.kda_head_dim
            shapes.update({
                "self_attn.q_proj.weight": (qkv, h),
                "self_attn.k_proj.weight": (qkv, h),
                "self_attn.v_proj.weight": (qkv, h),
                "self_attn.q_conv1d.weight": (qkv, 1, self.kda_conv_kernel_size),
                "self_attn.k_conv1d.weight": (qkv, 1, self.kda_conv_kernel_size),
                "self_attn.v_conv1d.weight": (qkv, 1, self.kda_conv_kernel_size),
                "self_attn.f_a_proj.weight": (self.kda_head_dim, h),
                "self_attn.f_b_proj.weight": (qkv, self.kda_head_dim),
                "self_attn.dt_bias": (qkv, ),
                "self_attn.A_log": (self.kda_num_heads, ),
                "self_attn.b_proj.weight": (self.kda_num_heads, h),
                "self_attn.g_a_proj.weight": (self.kda_head_dim, h),
                "self_attn.g_b_proj.weight": (qkv, self.kda_head_dim),
                "self_attn.o_norm.weight": (self.kda_head_dim, ),
                "self_attn.o_proj.weight": (h, qkv),
            })
        else:
            raise ImporterError(f"Unknown GLM-5.3 Flash decoder kind {kind!r}.")
        if self.quantized:
            # GLM's published checkpoint uses dynamic block FP8 selectively.
            # KDA projections, router/norm/indexer tensors and MLA's
            # unabsorbed kv_b projection remain BF16; only modules carrying a
            # matching ``weight_scale_inv`` enter the block-scaled ABI.
            fp8_weights = set()
            if mlp_kind == "dense":
                fp8_weights.update({
                    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
                })
            else:
                fp8_weights.update({
                    "mlp.experts.gate_up_proj", "mlp.experts.down_proj",
                    "mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight",
                    "mlp.shared_experts.down_proj.weight",
                })
            if kind == "mla_attention":
                fp8_weights.update({
                    "self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight",
                    "self_attn.kv_a_proj_with_mqa.weight", "self_attn.o_proj.weight",
                })
            for name, shape in tuple(shapes.items()):
                if name not in fp8_weights:
                    continue
                if len(shape) == 3:
                    shapes[name + "_scale_inv"] = (
                        shape[0], (shape[1] + self.weight_block_n - 1) // self.weight_block_n,
                        (shape[2] + self.weight_block_k - 1) // self.weight_block_k)
                elif len(shape) == 2:
                    shapes[name + "_scale_inv"] = (
                        (shape[0] + self.weight_block_n - 1) // self.weight_block_n,
                        (shape[1] + self.weight_block_k - 1) // self.weight_block_k)
                else:
                    raise ImporterError(
                        f"GLM FP8 weight {name!r} must be rank 2 or 3, got {len(shape)}.")
            return {
                name: tensor_type(
                    "float32" if name in {"self_attn.A_log", "self_attn.dt_bias",
                                             "hc_attn_base", "hc_attn_scale",
                                             "hc_ffn_base", "hc_ffn_scale",
                                             "mlp.gate.e_score_correction_bias"}
                    or name.endswith("_scale_inv")
                    else "float8_e4m3fn" if name in fp8_weights else "bfloat16",
                    shape)
                for name, shape in shapes.items()
            }
        return {
            name: tensor_type("float32" if name in {"self_attn.A_log", "self_attn.dt_bias",
                                                       "hc_attn_base", "hc_attn_scale",
                                                       "hc_ffn_base", "hc_ffn_scale",
                                                       "mlp.gate.e_score_correction_bias"}
                              else "bfloat16",
                              shape)
            for name, shape in shapes.items()
        }
