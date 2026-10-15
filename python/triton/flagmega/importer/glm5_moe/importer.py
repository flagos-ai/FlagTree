# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Supported GLM-5.3-Flash single-layer text decode subset.

Imports one ``deepseek_sparse_attention`` or sparse-MLP KDA layer past
``first_k_dense_replace``. It intentionally rejects native mHC/DSA/FP8 model
features until their full contracts are implemented. Weight names follow the
published Hugging Face checkpoint.
"""

from collections import Counter
from pathlib import Path

from triton.flagmega.errors import ImporterError
from triton.flagmega.importer.checkpoint import DirectoryCheckpoint
from triton.flagmega.importer.glm5_moe.config import Glm5MoeConfig
from triton.flagmega.importer.glm5_moe.decoder import (
    apply_hyper_connection,
    build_dense_mlp,
    build_hyper_connection,
    build_kda_attention,
    build_mla_attention,
    build_moe,
    linear,
    rms_norm,
)
from triton.flagmega.importer.source import attach_import_source_locations
from triton.flagmega.ir import (
    BankField,
    F,
    Module,
    ParameterBankSchema,
    TupleType,
    effect,
    tensor_type,
    verify_module,
)
from triton.flagmega.ir.ops.nn._dsa_indexer_state import DsaIndexerStateConfig
from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionStateConfig


def _kda_conv_weight(weights, config, *, name):
    """Assemble the depthwise conv weight as ``[3, heads, head_dim, kernel]``.

    The published checkpoint stores three independent ``[heads*head_dim, 1,
    kernel]`` tensors.  Each is reshaped in place to ``[1, heads, head_dim,
    kernel]`` and only the leading *part* axis is joined, so the head axis stays
    a single contiguous range per part — the property a head split needs.
    Flattening q|k|v onto one channel axis instead would interleave the parts.
    """

    parts = tuple(
        F.tensors.reshape(weights.pop(f"self_attn.{part}_conv1d.weight"),
                          (1, config.kda_num_heads, config.kda_head_dim, config.kda_conv_kernel_size),
                          name=f"{name}_{part}")
        for part in ("q", "k", "v"))
    return F.tensors.concat(*parts, axis=0, name=name)


class Glm5MoeImporter:
    """Import text inference for one GLM-5.3-Flash sparse-MLA MoE layer."""

    kind = "mla_attention"

    def __init__(self, checkpoint, *, layer=None, revision=None, block_size=256, num_blocks=16,
                 execution_phase="decode", num_tokens=1):
        self.checkpoint = DirectoryCheckpoint(checkpoint) if isinstance(checkpoint, (str, Path)) else checkpoint
        self.config = Glm5MoeConfig.parse(self.checkpoint.config)
        if execution_phase not in ("decode", "prefill"):
            raise ImporterError("GLM-5.3 Flash execution_phase must be decode or prefill.")
        if execution_phase != "decode" or num_tokens != 1:
            raise ImporterError("GLM-5.3 Flash importer currently requires decode with one token.")
        self.execution_phase = execution_phase
        self.num_tokens = num_tokens
        self.revision = revision
        c = self.config
        if layer is None:
            layer = next((index for index in range(c.num_hidden_layers)
                          if c.layer_types[index] == "deepseek_sparse_attention"
                          and c.mlp_layer_types[index] == "sparse"), None)
            if layer is None:
                raise ImporterError("GLM-5.3 Flash checkpoint has no sparse MLA layer to import.")
        if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < c.num_hidden_layers:
            raise ImporterError("GLM-5.3 Flash requested layer is out of range.")
        kind = c.layer_types[layer]
        if kind == "linear_attention":
            if c.mlp_layer_types[layer] != "sparse":
                raise ImporterError(
                    "GLM-5.3 Flash dense-MLP layers (0-2) are KDA-attention; the dense-MLP decoder "
                    "kind is not imported yet — pick a sparse-MLP layer past first_k_dense_replace.")
            self.kind = "linear_attention"
        elif kind == "deepseek_sparse_attention" and c.mlp_layer_types[layer] == "sparse":
            self.kind = "mla_attention"
        else:
            raise ImporterError(f"GLM-5.3 Flash layer {layer} kind {kind!r} is not importable.")
        self.layer = layer
        prefixes = tuple(prefix for prefix in ("model.language_model.", "model.")
                         if prefix + "embed_tokens.weight" in self.checkpoint.keys)
        if len(prefixes) != 1:
            raise ImporterError("GLM-5.3 Flash checkpoint must have one unambiguous text embedding prefix.")
        self.model_prefix = prefixes[0]
        self.paged_config = (PagedAttentionStateConfig(1, c.num_attention_heads, c.qk_nope_head_dim,
                                                       block_size=block_size, num_blocks=num_blocks)
                             if self.kind == "mla_attention" else None)

    def import_module(self):
        c = self.config
        importer = self
        metadata = {
            "architecture": "Glm5NextForConditionalGeneration",
            "model_type": "glm5_next_text",
            "mode": "decode-1",
            "execution_phase": self.execution_phase,
            "tokens_per_call": self.num_tokens,
            "scope": "text-decoder",
            "revision": self.revision,
            "model_prefix": self.model_prefix,
            "imported_layer_indices": (self.layer, ),
            "layer_types": (c.layer_types[self.layer], ),
            "hidden_size": c.hidden_size,
            "vocab_size": c.vocab_size,
            "num_experts": c.num_experts,
            "num_experts_per_tok": c.num_experts_per_tok,
            "routed_scaling_factor": c.routed_scaling_factor,
            "swiglu_limit": c.swiglu_limit,
            "numerical_contract": "nncase",
            "storage_dtype": "bfloat16",
            "kda_num_heads": c.kda_num_heads,
            "kda_head_dim": c.kda_head_dim,
            "kda_conv_kernel_size": c.kda_conv_kernel_size,
        }

        class Graph(Module):

            def weight_at(self, key, value_type, *, group=None):
                try:
                    info = importer.checkpoint.tensor_info(key)
                except KeyError as error:
                    raise ImporterError(
                        f"GLM-5.3 Flash checkpoint is missing required tensor {key!r}; "
                        "the supported importer contract uses packed BF16 layer tensors."
                    ) from error
                expected = tuple(dim.fixed_value for dim in value_type.shape)
                if info.shape != expected or info.dtype != value_type.dtype:
                    raise ImporterError(
                        f"Tensor {key!r} requires {value_type.dtype.value}{expected}; got {info.dtype.value}{info.shape}."
                    )
                return self.weight(key, value_type, key=key, source=info.source, id="w_" + key.replace(".", "_"),
                                   metadata={} if group is None else {"rdata_group": group})

            def decoder(self, state_types):
                name = f"decode_{importer.kind}"
                params = []

                def parameter(key, value_type):
                    node = self.input(key, value_type, id=f"{name}_{key.replace('.', '_')}",
                                      metadata={"function_parameter": name})
                    params.append(node)
                    return node

                hidden = parameter("hidden", tensor_type("bfloat16", (importer.num_tokens, c.hidden_size)))
                states = [parameter(f"state_{index}", state_type)
                          for index, state_type in enumerate(state_types)]
                types = c.weight_types(importer.kind)
                # Offline splits of the stacked expert bank and the per-head
                # [k, v] kv_b rows belong to main's constant island, outside
                # the shared decoder's runtime dataflow.
                del types["mlp.experts.gate_up_proj"]
                types["mlp.experts.gate_proj"] = tensor_type("bfloat16",
                                                             (c.num_experts, c.moe_intermediate_size, c.hidden_size))
                types["mlp.experts.up_proj"] = types["mlp.experts.gate_proj"]

                if importer.kind == "mla_attention":
                    del types["self_attn.kv_b_proj.weight"]
                    types["self_attn.k_b_proj"] = tensor_type("bfloat16",
                                                              (c.num_attention_heads * c.qk_nope_head_dim,
                                                               c.kv_lora_rank))
                    types["self_attn.v_b_proj"] = tensor_type("bfloat16", (c.num_attention_heads * c.v_head_dim,
                                                                           c.kv_lora_rank))
                else:
                    # The per-channel decay maps are offline weight transforms
                    # in main's constant island; the decoder consumes ready
                    # (1, heads, head_dim) FP32 maps with no broadcasts.
                    for conv_key in ("self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight",
                                     "self_attn.v_conv1d.weight"):
                        del types[conv_key]
                    del types["self_attn.dt_bias"]
                    del types["self_attn.A_log"]
                    types["self_attn.conv1d.weight"] = tensor_type(
                        "bfloat16", (3, c.kda_num_heads, c.kda_head_dim, c.kda_conv_kernel_size))
                    types["self_attn.dt_bias_map"] = tensor_type(
                        "float32", (1, c.kda_num_heads, c.kda_head_dim))
                    types["self_attn.decay_rate_map"] = tensor_type(
                        "float32", (1, c.kda_num_heads, c.kda_head_dim))
                weights = {key: parameter(key, value_type) for key, value_type in types.items()}
                normalized = rms_norm(hidden, weights["input_layernorm.weight"], c.epsilon, name=f"{name}_input_norm")
                if importer.kind == "mla_attention":
                    layer_id = parameter("layer_id", tensor_type("int32", ()))
                    advance = parameter("advance_sequence", tensor_type("bool", ()))
                    attention, updated, _, _ = build_mla_attention(normalized, states[0], layer_id, advance, weights, c,
                                                                    prefix=name)
                    outputs = (updated, )
                else:
                    attention, conv_out, rec_out = build_kda_attention(normalized, states[0], states[1], weights, c,
                                                                      prefix=name)
                    outputs = (conv_out, rec_out)
                residual = F.math.add(hidden, attention, name=f"{name}_attention_residual")
                normalized = rms_norm(residual, weights["post_attention_layernorm.weight"], c.epsilon,
                                      name=f"{name}_post_norm")
                output = F.math.add(residual, build_moe(normalized, weights, c, prefix=name), name=f"{name}_output")
                self.function(name, params, (output, *outputs),
                              attrs={"calling_convention": "device", "noinline": True, "reusable": True})
                return name, tuple(types)

            def forward(self):
                ids = self.input("input_ids", tensor_type("int32", (importer.num_tokens, )), id="input_ids")
                if importer.kind == "mla_attention":
                    paged = self.input("paged_attention_state", importer.paged_config.ref_type, id="paged_state")
                    state_args = [paged]
                    state_types = [paged.type]
                else:
                    conv = self.input(
                        "kda_conv_state",
                        tensor_type("bfloat16", (3, c.kda_num_heads, c.kda_head_dim,
                                                 c.kda_conv_kernel_size - 1)),
                        id="kda_conv_state")
                    rec = self.input(
                        "kda_recurrent_state",
                        tensor_type("float32", (1, c.kda_num_heads, c.kda_head_dim, c.kda_head_dim)),
                        id="kda_recurrent_state")
                    state_args = [conv, rec]
                    state_types = [conv.type, rec.type]
                name, keys = self.decoder(state_types)
                embedding = self.weight_at(importer.model_prefix + "embed_tokens.weight",
                                           tensor_type("bfloat16", (c.vocab_size, c.hidden_size)))
                hidden = F.nn.embedding(ids, embedding, padding_idx=c.padding_idx, name="token_embedding")
                weights = {
                    key: self.weight_at(f"{importer.model_prefix}layers.{importer.layer}.{key}", value_type)
                    for key, value_type in c.weight_types(importer.kind).items()
                }
                stacked = weights.pop("mlp.experts.gate_up_proj")
                for part, begin in (("gate", 0), ("up", c.moe_intermediate_size)):
                    weights[f"mlp.experts.{part}_proj"] = F.tensors.slice(
                        stacked, starts=(begin, ), ends=(begin + c.moe_intermediate_size, ), axes=(1, ),
                        name=f"layer_{importer.layer}_expert_{part}")

                if importer.kind == "linear_attention":
                    # Constant-island weight transforms: exp(A_log) broadcast to
                    # a per-channel decay map and the dt_bias row reshaped to
                    # head layout — the published exp(A_log)·(g + dt_bias)
                    # gate math with no weight-side broadcasts at runtime.
                    heads, dim = c.kda_num_heads, c.kda_head_dim
                    weights["self_attn.conv1d.weight"] = _kda_conv_weight(
                        weights, c, name=f"layer_{importer.layer}_qkv_conv_weight")
                    a_log = weights.pop("self_attn.A_log")
                    dt_bias = weights.pop("self_attn.dt_bias")
                    decay_rate = F.math.exp(a_log, name=f"layer_{importer.layer}_exp_a_log")
                    weights["self_attn.decay_rate_map"] = F.tensors.broadcast_to(
                        F.tensors.reshape(decay_rate, (1, heads, 1),
                                          name=f"layer_{importer.layer}_decay_col"),
                        shape=(1, heads, dim), name=f"layer_{importer.layer}_decay_rate_map")
                    weights["self_attn.dt_bias_map"] = F.tensors.reshape(
                        dt_bias, (1, heads, dim), name=f"layer_{importer.layer}_dt_bias_map")
                if importer.kind == "mla_attention":
                    # Regroup the checkpoint's per-head [k, v] kv_b rows into one
                    # all-k weight and one all-v weight; a pure row permutation of
                    # the same dot products, done offline in the constant island.
                    kv_b = weights.pop("self_attn.kv_b_proj.weight")
                    head_span = c.qk_nope_head_dim + c.v_head_dim
                    k_rows = F.tensors.concat(
                        *(F.tensors.slice(kv_b, starts=(head * head_span, ),
                                          ends=(head * head_span + c.qk_nope_head_dim, ), axes=(-2, ),
                                          name=f"layer_{importer.layer}_k_rows_{head}")
                          for head in range(c.num_attention_heads)),
                        axis=-2, name=f"layer_{importer.layer}_k_regrouped")
                    v_rows = F.tensors.concat(
                        *(F.tensors.slice(kv_b, starts=(head * head_span + c.qk_nope_head_dim, ),
                                          ends=((head + 1) * head_span, ), axes=(-2, ),
                                          name=f"layer_{importer.layer}_v_rows_{head}")
                          for head in range(c.num_attention_heads)),
                        axis=-2, name=f"layer_{importer.layer}_v_regrouped")
                    weights["self_attn.k_b_proj"] = k_rows
                    weights["self_attn.v_b_proj"] = v_rows
                if importer.kind == "mla_attention":
                    # Decoder parameter order: hidden, state, weights, layer_id, advance.
                    layer_id = F.builtin.scalar_const(tensor_type("int32", ()), 0, name="layer_state_id")
                    advance = F.builtin.scalar_const(tensor_type("bool", ()), True, name="advance_sequence")
                    args = [hidden, *state_args, *(weights[key] for key in keys), layer_id, advance]
                    result_type = TupleType((hidden.type, paged.type))
                else:
                    args = [hidden, *state_args, *(weights[key] for key in keys)]
                    result_type = TupleType((hidden.type, conv.type, rec.type))
                call = F.builtin.call(*args,
                                      callee=name,
                                      result_type=result_type,
                                      effect=effect("read_write", "paged_attention_kv_cache")
                                      if importer.kind == "mla_attention" else effect("pure"),
                                      name=f"layer_{importer.layer}_call",
                                      metadata={"layer_index": importer.layer, "state_layer_index": 0})
                if importer.kind == "mla_attention":
                    hidden, paged = F.tensors.get_items(call, 0, 1, name_prefix=f"layer_{importer.layer}")
                    state_outputs = (paged, )
                else:
                    hidden, conv, rec = F.tensors.get_items(call, 0, 1, 2, name_prefix=f"layer_{importer.layer}")
                    state_outputs = (conv, rec)
                norm_weight = self.weight_at(importer.model_prefix + "norm.weight",
                                             tensor_type("bfloat16", (c.hidden_size, )))
                hidden = rms_norm(hidden, norm_weight, c.epsilon, name="final_norm")
                lm_head = embedding if c.tie_word_embeddings else self.weight_at(
                    "lm_head.weight", tensor_type("bfloat16", (c.vocab_size, c.hidden_size)))
                logits = linear(hidden, lm_head, output_dtype="float32", name="logits")
                token = F.nn.greedy_sample(logits, name="next_token")
                self.function("main", (ids, *state_args), (logits, token, *state_outputs))

        graph = Graph(dialect="high_level", stage="imported", entry="main", metadata=metadata).build()
        return verify_module(
            attach_import_source_locations(graph, architecture=metadata["architecture"], revision=self.revision))


class Glm5MoeModelImporter:
    """Import the complete supported BF16 GLM text decoder.

    The layer schedule and state ownership are derived from ``config.json``:
    KDA layers own independent convolution/recurrent tensor states, while all
    MLA layers share one paged-cache reference with a local MLA layer index.
    This keeps the model ABI independent of the published 45-layer cadence and
    also makes reduced fixtures exercise the same graph shape.
    """

    def __init__(self, checkpoint, *, revision=None, block_size=256, num_blocks=16,
                 execution_phase="decode", num_tokens=1):
        self.checkpoint = DirectoryCheckpoint(checkpoint) if isinstance(checkpoint, (str, Path)) else checkpoint
        self.config = Glm5MoeConfig.parse(self.checkpoint.config, allow_mhc=True)
        if execution_phase not in ("decode", "prefill"):
            raise ImporterError("GLM-5.3 Flash execution_phase must be decode or prefill.")
        if type(num_tokens) is not int or num_tokens <= 0 or execution_phase == "decode" and num_tokens != 1:
            raise ImporterError("GLM-5.3 Flash decode requires one token; prefill requires a positive token count.")
        self.execution_phase = execution_phase
        self.num_tokens = num_tokens
        self.revision = revision
        c = self.config
        self.layers = tuple(range(c.num_hidden_layers))
        self.mla_layers = tuple(index for index in self.layers if c.layer_types[index] == "deepseek_sparse_attention")
        self.kda_layers = tuple(index for index in self.layers if c.layer_types[index] == "linear_attention")
        self.full_indexer_layers = tuple(index for index in self.mla_layers
                                        if c.indexer_types and c.indexer_types[index] == "full")
        self.dsa_config = (DsaIndexerStateConfig(
            self.full_indexer_layers, block_size * num_blocks, c.index_head_dim,
            c.index_topk + (c.index_kpool - 1 if c.index_kpool_always_select_tail else 0))
                           if c.indexer_types else None)
        prefixes = tuple(prefix for prefix in ("model.language_model.", "model.")
                         if prefix + "embed_tokens.weight" in self.checkpoint.keys)
        if len(prefixes) != 1:
            raise ImporterError("GLM-5.3 Flash checkpoint must have one unambiguous text embedding prefix.")
        self.model_prefix = prefixes[0]
        self.paged_config = PagedAttentionStateConfig(
            max(1, len(self.mla_layers)), c.num_attention_heads, c.qk_nope_head_dim,
            block_size=block_size, num_blocks=num_blocks)

    def parameter_bank_schema(self, layers, keys, *, name):
        """Build a verified checkpoint-row schema for one repeated layer family.

        This helper is deliberately metadata-first: it rejects shape/dtype
        drift before a future structured importer creates a ``For`` body. The
        current layer families with mutable state use separate state contracts
        and must opt in explicitly rather than being silently flattened into a
        bank.
        """
        layers = tuple(int(layer) for layer in layers)
        keys = tuple(str(key) for key in keys)
        if not layers or not keys or any(layer not in self.layers for layer in layers):
            raise ImporterError("Parameter bank requires nonempty valid layer and field lists.")
        fields = []
        rows = []
        for key in keys:
            infos = []
            row = []
            for layer in layers:
                full_key = f"{self.model_prefix}layers.{layer}.{key}"
                try:
                    info = self.checkpoint.tensor_info(full_key)
                except KeyError as error:
                    raise ImporterError(f"Parameter bank is missing {full_key!r}.") from error
                infos.append(info)
                row.append(full_key)
            first = infos[0]
            if any(info.shape != first.shape or info.dtype != first.dtype for info in infos[1:]):
                raise ImporterError(f"Parameter bank field {key!r} changes shape or dtype across rows.")
            fields.append(BankField(key, tensor_type(first.dtype.value, first.shape), "checkpoint"))
            rows.append(tuple(row))
        # The schema is consumed by the importer and is part of the module's
        # semantic metadata; constructing it here never loads tensor payloads.
        return ParameterBankSchema(name, tuple(fields)), tuple(zip(*rows, strict=True))

    def import_module(self):
        c = self.config
        importer = self
        metadata = {
            "architecture": "Glm5NextForConditionalGeneration",
            "model_type": "glm5_next_text",
            "mode": "decode-1" if self.execution_phase == "decode" else "prefill",
            "execution_phase": self.execution_phase,
            "tokens_per_call": self.num_tokens,
            "scope": "text-decoder",
            "revision": self.revision,
            "model_prefix": self.model_prefix,
            "num_hidden_layers": c.num_hidden_layers,
            "source_num_hidden_layers": c.num_hidden_layers,
            "mhc": c.mhc,
            "hc_mult": c.hc_mult,
            "hc_sinkhorn_iters": c.hc_sinkhorn_iters,
            "imported_layer_indices": importer.layers,
            "layer_types": c.layer_types,
            "mlp_layer_types": c.mlp_layer_types,
            "hidden_size": c.hidden_size,
            "vocab_size": c.vocab_size,
            "num_experts": c.num_experts,
            "num_experts_per_tok": c.num_experts_per_tok,
            "routed_scaling_factor": c.routed_scaling_factor,
            "swiglu_limit": c.swiglu_limit,
            "numerical_contract": "nncase",
            "mla_layer_indices": importer.mla_layers,
            "kda_layer_indices": importer.kda_layers,
            "dsa_indexer_types": c.indexer_types,
            "dsa_full_indexer_layers": importer.full_indexer_layers,
            "dsa_index_topk": c.index_topk,
            "dsa_index_kpool": c.index_kpool,
            "dsa_index_output_width": (c.index_topk + c.index_kpool - 1
                                        if c.indexer_types and c.index_kpool_always_select_tail else c.index_topk),
            "kda_num_heads": c.kda_num_heads,
            "kda_head_dim": c.kda_head_dim,
            "kda_conv_kernel_size": c.kda_conv_kernel_size,
        }

        class Graph(Module):
            def weight_at(self, key, value_type, *, group=None):
                try:
                    info = importer.checkpoint.tensor_info(key)
                except KeyError as error:
                    raise ImporterError(f"GLM-5.3 Flash checkpoint is missing required tensor {key!r}.") from error
                expected = tuple(dim.fixed_value for dim in value_type.shape)
                if info.shape != expected or info.dtype != value_type.dtype:
                    raise ImporterError(
                        f"Tensor {key!r} requires {value_type.dtype.value}{expected}; "
                        f"got {info.dtype.value}{info.shape}."
                    )
                return self.weight(key, value_type, key=key, source=info.source,
                                   id="w_" + key.replace(".", "_"),
                                   metadata={} if group is None else {"rdata_group": group})

            def decoder(self, kind, mlp_kind, state_types, indexer_kind=None):
                decoder_kind = "mla_attention" if kind == "deepseek_sparse_attention" else kind
                name = f"decode_{decoder_kind}_{mlp_kind}"
                params = []

                def parameter(key, value_type):
                    node = self.input(key, value_type, id=f"{name}_{key.replace('.', '_')}",
                                      metadata={"function_parameter": name})
                    params.append(node)
                    return node

                hidden_shape = ((importer.num_tokens, c.hc_mult, c.hidden_size)
                                if c.mhc else (importer.num_tokens, c.hidden_size))
                hidden = parameter("hidden", tensor_type("bfloat16", hidden_shape))
                states = [parameter(f"state_{index}", state_type) for index, state_type in enumerate(state_types)]
                topk = None
                if decoder_kind == "mla_attention" and c.indexer_types and indexer_kind == "shared":
                    output_width = c.index_topk + (c.index_kpool - 1 if c.index_kpool_always_select_tail else 0)
                    topk = parameter("topk", tensor_type("int32", (importer.num_tokens, output_width)))
                types = c.weight_types(decoder_kind, mlp_kind, indexer_type=indexer_kind)
                if mlp_kind == "sparse":
                    del types["mlp.experts.gate_up_proj"]
                    if c.quantized:
                        del types["mlp.experts.gate_up_proj_scale_inv"]
                    expert_dtype = "float8_e4m3fn" if c.quantized else "bfloat16"
                    types["mlp.experts.gate_proj"] = tensor_type(
                        expert_dtype, (c.num_experts, c.moe_intermediate_size, c.hidden_size))
                    types["mlp.experts.up_proj"] = types["mlp.experts.gate_proj"]
                    if c.quantized:
                        types["mlp.experts.down_proj"] = tensor_type(
                            expert_dtype, (c.num_experts, c.hidden_size, c.moe_intermediate_size))
                        types["mlp.experts.gate_proj_scale_inv"] = tensor_type(
                            "float32", (c.num_experts,
                                         (c.moe_intermediate_size + c.weight_block_n - 1) // c.weight_block_n,
                                         (c.hidden_size + c.weight_block_k - 1) // c.weight_block_k))
                        types["mlp.experts.up_proj_scale_inv"] = types["mlp.experts.gate_proj_scale_inv"]
                        types["mlp.experts.down_proj_scale_inv"] = tensor_type(
                            "float32", (c.num_experts,
                                         (c.hidden_size + c.weight_block_n - 1) // c.weight_block_n,
                                         (c.moe_intermediate_size + c.weight_block_k - 1) // c.weight_block_k))
                if decoder_kind == "mla_attention":
                    if not c.quantized:
                        del types["self_attn.kv_b_proj.weight"]
                        types["self_attn.k_b_proj"] = tensor_type(
                            "bfloat16", (c.num_attention_heads * c.qk_nope_head_dim, c.kv_lora_rank))
                        types["self_attn.v_b_proj"] = tensor_type(
                            "bfloat16", (c.num_attention_heads * c.v_head_dim, c.kv_lora_rank))
                else:
                    for conv_key in ("self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight",
                                     "self_attn.v_conv1d.weight"):
                        del types[conv_key]
                    del types["self_attn.dt_bias"]
                    del types["self_attn.A_log"]
                    types["self_attn.conv1d.weight"] = tensor_type(
                        "bfloat16", (3, c.kda_num_heads, c.kda_head_dim, c.kda_conv_kernel_size))
                    types["self_attn.dt_bias_map"] = tensor_type(
                        "float32", (1, c.kda_num_heads, c.kda_head_dim))
                    types["self_attn.decay_rate_map"] = tensor_type(
                        "float32", (1, c.kda_num_heads, c.kda_head_dim))
                weights = {key: parameter(key, value_type) for key, value_type in types.items()}
                if c.mhc:
                    residual = hidden
                    attn_post, attn_comb, collapsed = build_hyper_connection(
                        hidden, weights, c, stage="attn", prefix=name)
                    normalized = rms_norm(collapsed, weights["input_layernorm.weight"], c.epsilon,
                                          name=f"{name}_input_norm")
                else:
                    residual = hidden
                    normalized = rms_norm(hidden, weights["input_layernorm.weight"], c.epsilon,
                                          name=f"{name}_input_norm")
                if decoder_kind == "mla_attention":
                    layer_id = parameter("layer_id", tensor_type("int32", ()))
                    dsa_layer_id = (parameter("dsa_layer_id", tensor_type("int32", ()))
                                   if c.indexer_types else layer_id)
                    advance = parameter("advance_sequence", tensor_type("bool", ()))
                    dsa_advance = (parameter("dsa_advance_sequence", tensor_type("bool", ()))
                                   if c.indexer_types else advance)
                    attention, updated, _updated_dsa, output_topk = build_mla_attention(
                        normalized, states[0], layer_id, advance, weights, c, prefix=name,
                        dsa_state=states[1] if c.indexer_types else None,
                        dsa_layer_id=dsa_layer_id, dsa_advance=dsa_advance, topk=topk,
                        indexer_kind=indexer_kind)
                    state_outputs = ((updated, output_topk)
                                     if c.indexer_types else (updated, ))
                else:
                    attention, conv_out, rec_out = build_kda_attention(
                        normalized, states[0], states[1], weights, c, prefix=name)
                    state_outputs = (conv_out, rec_out)
                if c.mhc:
                    hidden = apply_hyper_connection(attention, residual, attn_post, attn_comb, c,
                                                    stage="attn", prefix=name)
                    residual = hidden
                    ffn_post, ffn_comb, collapsed = build_hyper_connection(
                        hidden, weights, c, stage="ffn", prefix=name)
                    normalized = rms_norm(collapsed, weights["post_attention_layernorm.weight"], c.epsilon,
                                          name=f"{name}_post_norm")
                else:
                    # Keep the projection as the first operand so the
                    # AddNormStats lowering can preserve its partial-sum ABI
                    # when block-scaled FP8 matmuls are present.
                    residual = F.math.add(attention, hidden, name=f"{name}_attention_residual")
                    normalized = rms_norm(residual, weights["post_attention_layernorm.weight"], c.epsilon,
                                          name=f"{name}_post_norm")
                if mlp_kind == "dense":
                    mlp = build_dense_mlp(normalized, weights, c, prefix=name)
                else:
                    mlp = build_moe(normalized, weights, c, prefix=name)
                output = (apply_hyper_connection(mlp, residual, ffn_post, ffn_comb, c,
                                                  stage="ffn", prefix=name)
                          if c.mhc else F.math.add(residual, mlp, name=f"{name}_output"))
                self.function(name, params, (output, *state_outputs),
                              attrs={"calling_convention": "device", "noinline": True, "reusable": True})
                return name, tuple(types)

            def forward(self):
                ids = self.input("input_ids", tensor_type("int32", (importer.num_tokens,)), id="input_ids")
                kda_states = {}
                state_input_ids = {}
                for layer in importer.kda_layers:
                    kda_states[layer] = (
                        self.input(f"kda_conv_state_{layer}",
                                   tensor_type("bfloat16", (3, c.kda_num_heads, c.kda_head_dim,
                                                            c.kda_conv_kernel_size - 1)),
                                   id=f"kda_conv_state_{layer}"),
                        self.input(f"kda_recurrent_state_{layer}",
                                   tensor_type("float32", (1, c.kda_num_heads, c.kda_head_dim, c.kda_head_dim)),
                        id=f"kda_recurrent_state_{layer}"),
                    )
                    state_input_ids[layer] = tuple(value.id for value in kda_states[layer])
                entry_kda_states = tuple(
                    state for layer in importer.kda_layers for state in kda_states[layer]
                )
                entry_paged = paged = self.input("paged_attention_state", importer.paged_config.ref_type,
                                                 id="paged_state")
                entry_dsa = dsa = (self.input("dsa_indexer_state", importer.dsa_config.ref_type,
                                               id="dsa_state") if importer.dsa_config else None)
                embedding = self.weight_at(importer.model_prefix + "embed_tokens.weight",
                                           tensor_type("bfloat16", (c.vocab_size, c.hidden_size)))
                hidden = F.nn.embedding(ids, embedding, padding_idx=c.padding_idx, name="token_embedding")
                if c.mhc:
                    hidden = F.tensors.broadcast_to(
                        F.tensors.reshape(hidden, (importer.num_tokens, 1, c.hidden_size),
                                          name="initial_stream"),
                        shape=(importer.num_tokens, c.hc_mult, c.hidden_size),
                        name="initial_streams")
                decoder_keys = {}
                category_counts = Counter(
                    (c.layer_types[layer], c.mlp_layer_types[layer],
                     c.indexer_types[layer] if c.layer_types[layer] == "deepseek_sparse_attention" and c.indexer_types else None)
                    for layer in importer.layers)
                category_indices = Counter()
                for kind in ("linear_attention", "deepseek_sparse_attention"):
                    for mlp_kind in ("dense", "sparse"):
                        indexer_kinds = (("full", "shared") if kind == "deepseek_sparse_attention" and c.indexer_types
                                         else (None,))
                        for indexer_kind in indexer_kinds:
                            selected = [layer for layer in importer.layers
                                        if c.layer_types[layer] == kind and c.mlp_layer_types[layer] == mlp_kind
                                        and (indexer_kind is None or c.indexer_types[layer] == indexer_kind)]
                            if not selected:
                                continue
                            if kind == "linear_attention":
                                state_types = [kda_states[selected[0]][0].type, kda_states[selected[0]][1].type]
                            else:
                                state_types = [paged.type] + ([dsa.type] if importer.dsa_config else [])
                            decoder_keys[(kind, mlp_kind, indexer_kind)] = self.decoder(
                                kind, mlp_kind, state_types, indexer_kind=indexer_kind)
                mla_local = 0
                topk = (F.builtin.splat_const(
                    tensor_type("int32", (importer.num_tokens, c.index_topk + c.index_kpool - 1
                                           if c.indexer_types and c.index_kpool_always_select_tail
                                           else c.index_topk)), -1, name="initial_dsa_topk")
                        if c.indexer_types and "shared" in c.indexer_types else None)
                for layer in importer.layers:
                    kind = c.layer_types[layer]
                    mlp_kind = c.mlp_layer_types[layer]
                    indexer_kind = (c.indexer_types[layer]
                                     if kind == "deepseek_sparse_attention" and c.indexer_types else None)
                    name, keys = decoder_keys[(kind, mlp_kind, indexer_kind)]
                    contract_kind = "mla_attention" if kind == "deepseek_sparse_attention" else kind
                    types = c.weight_types(contract_kind, mlp_kind, indexer_type=indexer_kind)
                    category_key = (kind, mlp_kind, indexer_kind)
                    group_base = {"index": category_indices[category_key],
                                  "count": category_counts[category_key]}
                    layer_prefix = f"{importer.model_prefix}layers.{layer}."
                    packed_experts = layer_prefix + "mlp.experts.gate_up_proj" in importer.checkpoint.keys
                    weights = {
                        key: self.weight_at(
                            layer_prefix + key, value_type,
                            group={"name": f"decoder.{kind}.{mlp_kind}.{key}", **group_base})
                        for key, value_type in types.items()
                        if key not in ("mlp.experts.gate_up_proj", "mlp.experts.gate_up_proj_scale_inv",
                                        "mlp.experts.down_proj", "mlp.experts.down_proj_scale_inv") or packed_experts
                    }
                    category_indices[category_key] += 1
                    if mlp_kind == "sparse":
                        if packed_experts:
                            stacked = weights.pop("mlp.experts.gate_up_proj")
                            stacked_scale = (weights.pop("mlp.experts.gate_up_proj_scale_inv")
                                            if c.quantized else None)
                            for part, begin in (("gate", 0), ("up", c.moe_intermediate_size)):
                                weights[f"mlp.experts.{part}_proj"] = F.tensors.slice(
                                    stacked, starts=(begin,), ends=(begin + c.moe_intermediate_size,), axes=(1,),
                                    name=f"layer_{layer}_expert_{part}")
                                if stacked_scale is not None:
                                    scale_begin = begin // c.weight_block_n
                                    scale_end = (begin + c.moe_intermediate_size + c.weight_block_n - 1) // c.weight_block_n
                                    weights[f"mlp.experts.{part}_proj_scale_inv"] = F.tensors.slice(
                                        stacked_scale, starts=(scale_begin,), ends=(scale_end,), axes=(1,),
                                        name=f"layer_{layer}_expert_{part}_scale")
                        else:
                            # Published GLM-5.3-Flash shards store one tensor
                            # per expert.  Stack only the logical expert axis;
                            # the decoder and routing ABI remain independent of
                            # the checkpoint serialization layout.
                            for part, shape in (
                                ("gate", (c.moe_intermediate_size, c.hidden_size)),
                                ("up", (c.moe_intermediate_size, c.hidden_size)),
                                ("down", (c.hidden_size, c.moe_intermediate_size)),
                            ):
                                expert_dtype = "float8_e4m3fn" if c.quantized else "bfloat16"
                                parts = [F.tensors.reshape(self.weight_at(
                                    f"{layer_prefix}mlp.experts.{expert}.{part}_proj.weight",
                                    tensor_type(expert_dtype, shape),
                                    group={"name": f"decoder.{kind}.{mlp_kind}.experts.{part}", **group_base}),
                                    (1, *shape), name=f"layer_{layer}_expert_{expert}_{part}_axis")
                                         for expert in range(c.num_experts)]
                                weights[f"mlp.experts.{part}_proj"] = F.tensors.concat(
                                    *parts, axis=0, name=f"layer_{layer}_expert_{part}_stacked")
                                if c.quantized:
                                    scale_shape = (
                                        (c.moe_intermediate_size + c.weight_block_n - 1) // c.weight_block_n,
                                        (c.hidden_size + c.weight_block_k - 1) // c.weight_block_k,
                                    ) if part != "down" else (
                                        (c.hidden_size + c.weight_block_n - 1) // c.weight_block_n,
                                        (c.moe_intermediate_size + c.weight_block_k - 1) // c.weight_block_k,
                                    )
                                    scale_parts = [F.tensors.reshape(self.weight_at(
                                        f"{layer_prefix}mlp.experts.{expert}.{part}_proj.weight_scale_inv",
                                        tensor_type("float32", scale_shape),
                                        group={"name": f"decoder.{kind}.{mlp_kind}.experts.{part}.scale", **group_base}),
                                        (1, *scale_shape),
                                        name=f"layer_{layer}_expert_{expert}_{part}_scale_axis")
                                                   for expert in range(c.num_experts)]
                                    weights[f"mlp.experts.{part}_proj_scale_inv"] = F.tensors.concat(
                                        *scale_parts, axis=0,
                                        name=f"layer_{layer}_expert_{part}_scale_stacked")
                    if kind == "linear_attention":
                        heads, dim = c.kda_num_heads, c.kda_head_dim
                        weights["self_attn.conv1d.weight"] = _kda_conv_weight(
                            weights, c, name=f"layer_{layer}_qkv_conv_weight")
                        a_log = weights.pop("self_attn.A_log")
                        dt_bias = weights.pop("self_attn.dt_bias")
                        weights["self_attn.decay_rate_map"] = F.tensors.broadcast_to(
                            F.tensors.reshape(F.math.exp(a_log, name=f"layer_{layer}_exp_a_log"), (1, heads, 1),
                                              name=f"layer_{layer}_decay_col"),
                            shape=(1, heads, dim), name=f"layer_{layer}_decay_rate_map")
                        weights["self_attn.dt_bias_map"] = F.tensors.reshape(
                            dt_bias, (1, heads, dim), name=f"layer_{layer}_dt_bias_map")
                    else:
                        if not c.quantized:
                            kv_b = weights.pop("self_attn.kv_b_proj.weight")
                            span = c.qk_nope_head_dim + c.v_head_dim
                            weights["self_attn.k_b_proj"] = F.tensors.concat(
                                *(F.tensors.slice(kv_b, starts=(head * span,),
                                                   ends=(head * span + c.qk_nope_head_dim,), axes=(-2,))
                                  for head in range(c.num_attention_heads)), axis=-2,
                                name=f"layer_{layer}_k_regrouped")
                            weights["self_attn.v_b_proj"] = F.tensors.concat(
                                *(F.tensors.slice(kv_b, starts=(head * span + c.qk_nope_head_dim,),
                                                   ends=((head + 1) * span,), axes=(-2,))
                                  for head in range(c.num_attention_heads)), axis=-2,
                                name=f"layer_{layer}_v_regrouped")
                    if kind == "linear_attention":
                        state_args = list(kda_states[layer])
                        result_type = TupleType((hidden.type, *[value.type for value in state_args]))
                        effect_kind = effect("read_write", "kda_conv_state")
                    else:
                        state_args = [paged] + ([dsa] if importer.dsa_config else [])
                        layer_id = F.builtin.scalar_const(tensor_type("int32", ()), mla_local,
                                                          name=f"layer_{layer}_state_id")
                        dsa_layer_id = F.builtin.scalar_const(tensor_type("int32", ()), layer,
                                                              name=f"layer_{layer}_dsa_state_id")
                        advance = F.builtin.scalar_const(tensor_type("bool", ()),
                                                         mla_local + 1 == len(importer.mla_layers),
                                                         name=f"layer_{layer}_advance_sequence")
                        dsa_advance = F.builtin.scalar_const(
                            tensor_type("bool", ()),
                            bool(importer.full_indexer_layers and layer == importer.full_indexer_layers[-1]),
                            name=f"layer_{layer}_dsa_advance_sequence")
                        mla_local += 1
                        if importer.dsa_config:
                            dsa_topk_type = tensor_type(
                                "int32", (importer.num_tokens, c.index_topk + c.index_kpool - 1
                                           if c.index_kpool_always_select_tail else c.index_topk))
                            result_type = TupleType((hidden.type, paged.type, dsa_topk_type))
                        else:
                            result_type = TupleType((hidden.type, paged.type))
                        effect_kind = effect("read_write", "paged_attention_kv_cache")
                    args = [hidden, *state_args]
                    if kind == "deepseek_sparse_attention" and importer.dsa_config and indexer_kind == "shared":
                        args.append(topk)
                    args.extend(weights[key] for key in keys)
                    if kind == "deepseek_sparse_attention":
                        args.extend((layer_id, dsa_layer_id, advance) if importer.dsa_config
                                    else (layer_id, advance))
                        if importer.dsa_config:
                            args.append(dsa_advance)
                    call = F.builtin.call(*args, callee=name, result_type=result_type, effect=effect_kind,
                                          name=f"layer_{layer}_call",
                                          metadata={"layer_index": layer, "state_layer_index": layer})
                    items = F.tensors.get_items(call, *range(len(result_type.fields)), name_prefix=f"layer_{layer}")
                    hidden = items[0]
                    if kind == "linear_attention":
                        kda_states[layer] = (items[1], items[2])
                    else:
                        paged = items[1]
                        if importer.dsa_config:
                            topk = items[2]
                if c.mhc:
                    hidden = F.math.mul(
                        F.math.reduce_sum(hidden, axes=(1,), keep_dims=False, name="collapse_head_streams"),
                        F.builtin.splat_const(tensor_type("bfloat16", (importer.num_tokens, c.hidden_size)),
                                             1.0 / c.hc_mult),
                        name="collapse_head_mean")
                norm_weight = self.weight_at(importer.model_prefix + "norm.weight",
                                             tensor_type("bfloat16", (c.hidden_size,)))
                hidden = rms_norm(hidden, norm_weight, c.epsilon, name="final_norm")
                lm_head = embedding if c.tie_word_embeddings else self.weight_at(
                    "lm_head.weight", tensor_type("bfloat16", (c.vocab_size, c.hidden_size)))
                logits = linear(hidden, lm_head, output_dtype="float32", name="logits")
                token = F.nn.greedy_sample(logits, name="next_token")
                params = [ids, *entry_kda_states]
                params.append(entry_paged)
                if importer.dsa_config:
                    params.append(entry_dsa)
                outputs = [logits, token]
                for layer in importer.kda_layers:
                    outputs.extend(kda_states[layer])
                outputs.append(paged)
                self.builder.metadata["composite_abi"] = {
                    "schema": "flagmega.composite-abi/v1",
                    "input_id": ids.id,
                    "logits_output": logits.id,
                    "token_output": token.id,
                    "tensor_states": tuple({
                        "state_id": layer,
                        "input": state_input_ids[layer],
                        "output": tuple(value.id for value in kda_states[layer]),
                    } for layer in importer.kda_layers),
                    "paged_state": {
                        "input": entry_paged.id,
                        "output": paged.id,
                    },
                    **({"dsa_state": {"input": entry_dsa.id}}
                       if importer.dsa_config else {}),
                }
                self.function("main", params, outputs)

        graph = Graph(dialect="high_level", stage="imported", entry="main", metadata=metadata).build()
        return verify_module(attach_import_source_locations(
            graph, architecture=metadata["architecture"], revision=self.revision))


__all__ = ["Glm5MoeImporter", "Glm5MoeModelImporter"]
