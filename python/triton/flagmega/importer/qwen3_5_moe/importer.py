# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Complete BF16 hybrid text decode with one reusable function per layer kind."""

from collections import Counter
from pathlib import Path

from triton.flagmega.errors import ImporterError
from triton.flagmega.importer.checkpoint import DirectoryCheckpoint
from triton.flagmega.importer.source import attach_import_source_locations
from triton.flagmega.ir import F, Module, TupleType, effect, tensor_type, verify_module
from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetStateConfig
from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionStateConfig
from triton.flagmega.importer.qwen3_5_moe.config import Qwen35MoeConfig
from triton.flagmega.importer.qwen3_5_moe.decoder import build_full_attention, build_linear_attention, build_moe, linear, rms_norm


class Qwen35MoeImporter:
    """Import text inference only; vision and speculative MTP are separate models."""

    def __init__(self, checkpoint, *, layer=None, revision=None, block_size=256, num_blocks=16,
                 execution_phase="decode", num_tokens=1):
        self.checkpoint = DirectoryCheckpoint(checkpoint) if isinstance(checkpoint, (str, Path)) else checkpoint
        self.config = Qwen35MoeConfig.parse(self.checkpoint.config)
        if execution_phase not in ("decode", "prefill"):
            raise ImporterError("Qwen3.5 MoE execution_phase must be decode or prefill; it is not inferred from state.")
        if type(num_tokens) is not int or num_tokens <= 0 or execution_phase == "decode" and num_tokens != 1:
            raise ImporterError("Token count must be a positive integer; decode requires exactly one token.")
        self.execution_phase = execution_phase
        self.num_tokens = num_tokens
        c = self.config
        if layer is not None and (isinstance(layer, bool) or not isinstance(layer, int)
                                  or not 0 <= layer < c.num_hidden_layers):
            raise ImporterError("Qwen3.5 MoE requested layer is out of range.")
        self.layers = tuple(range(c.num_hidden_layers)) if layer is None else (layer, )
        self.revision = revision
        prefixes = tuple(prefix for prefix in ("model.language_model.", "model.")
                         if prefix + "embed_tokens.weight" in self.checkpoint.keys)
        if len(prefixes) != 1:
            raise ImporterError("Qwen3.5 MoE checkpoint must have one unambiguous text embedding prefix.")
        self.model_prefix = prefixes[0]
        counts = Counter(c.layer_types[index] for index in self.layers)
        self.gdn_config = GatedDeltaNetStateConfig(max(1,
                                                       counts["linear_attention"]), c.num_key_heads, c.num_value_heads,
                                                   c.key_head_dim, c.value_head_dim, c.conv_kernel_size, c.hidden_size)
        self.paged_config = PagedAttentionStateConfig(max(1, counts["full_attention"]), c.num_key_value_heads,
                                                      c.head_dim, block_size=block_size, num_blocks=num_blocks)

    def import_module(self):
        c = self.config
        importer = self

        class Graph(Module):

            def weight_at(self, key, value_type, *, group=None):
                info = importer.checkpoint.tensor_info(key)
                expected = tuple(dim.fixed_value for dim in value_type.shape)
                if info.shape != expected or info.dtype != value_type.dtype:
                    raise ImporterError(
                        f"Tensor {key!r} requires {value_type.dtype.value}{expected}; got {info.dtype.value}{info.shape}."
                    )
                return self.weight(key, value_type, key=key, source=info.source, id="w_" + key.replace(".", "_"),
                                   metadata={} if group is None else {"rdata_group": group})

            def decoder(self, kind, state_type, rotary_type):
                name = importer.execution_phase + ("_linear" if kind == "linear_attention" else "_attention")
                params = []

                def parameter(key, value_type):
                    node = self.input(key, value_type, id=f"{name}_{key.replace('.', '_')}",
                                      metadata={"function_parameter": name})
                    params.append(node)
                    return node

                hidden = parameter("hidden", tensor_type("bfloat16", (importer.num_tokens, c.hidden_size)))
                state = parameter("state", state_type)
                layer_id = parameter("layer_id", tensor_type("int32", ()))
                advance = cosine = sine = None
                if kind == "full_attention":
                    advance = parameter("advance_sequence", tensor_type("bool", ()))
                    cosine, sine = (parameter(key, rotary_type) for key in ("rotary_cos", "rotary_sin"))
                types = c.weight_types(kind)
                # Offline split of the stacked bank belongs to main's constant
                # island, outside the shared decoder's runtime dataflow.
                del types["mlp.experts.gate_up_proj"]
                types["mlp.experts.gate_proj"] = tensor_type("bfloat16",
                                                             (c.num_experts, c.intermediate_size, c.hidden_size))
                types["mlp.experts.up_proj"] = types["mlp.experts.gate_proj"]
                weights = {key: parameter(key, value_type) for key, value_type in types.items()}
                normalized = rms_norm(hidden, weights["input_layernorm.weight"], c.epsilon, name=f"{name}_input_norm")
                if kind == "linear_attention":
                    attention = build_linear_attention(normalized, state, layer_id, weights, c, prefix=name)
                    updated = state  # The one-layer view writes into this owner.
                else:
                    attention, updated = build_full_attention(normalized, state, layer_id, advance, cosine, sine,
                                                              weights, c, prefix=name)
                residual = F.math.add(hidden, attention, name=f"{name}_attention_residual")
                normalized = rms_norm(residual, weights["post_attention_layernorm.weight"], c.epsilon,
                                      name=f"{name}_post_norm")
                output = F.math.add(residual, build_moe(normalized, weights, c, prefix=name), name=f"{name}_output")
                self.function(name, params, (output, updated),
                              attrs={"calling_convention": "device", "noinline": True, "reusable": True})
                return name, tuple(types)

            def forward(self):
                ids = self.input("input_ids", tensor_type("int32", (importer.num_tokens, )), id="input_ids")
                entry_gdn = gdn = self.input("gated_delta_net_state", importer.gdn_config.ref_type, id="gdn_state")
                entry_paged = paged = self.input("paged_attention_state", importer.paged_config.ref_type,
                                                 id="paged_state")
                embedding = self.weight_at(importer.model_prefix + "embed_tokens.weight",
                                           tensor_type("bfloat16", (c.vocab_size, c.hidden_size)))
                hidden = F.nn.embedding(ids, embedding, padding_idx=c.padding_idx, name="token_embedding")
                positions = F.nn.rotary_embedding(hidden, paged, head_dim=c.rotary_dim, theta=c.rope_theta,
                                                  name="position_embeddings")
                cosine, sine = F.tensors.get_items(positions, 0, 1, name_prefix="main_rotary")
                counts = Counter(c.layer_types[index] for index in importer.layers)
                decoders = {
                    kind: self.decoder(kind, gdn.type if kind == "linear_attention" else paged.type, cosine.type)
                    for kind in dict.fromkeys(c.layer_types[index] for index in importer.layers)
                }
                indices = Counter()
                for layer in importer.layers:
                    kind = c.layer_types[layer]
                    local_layer = indices[kind]
                    indices[kind] += 1
                    name, keys = decoders[kind]
                    weights = {
                        key:
                        self.weight_at(
                            f"{importer.model_prefix}layers.{layer}.{key}", value_type,
                            group={"name": f"decoder.{kind}.{key}", "index": local_layer, "count": counts[kind]})
                        for key, value_type in c.weight_types(kind).items()
                    }
                    stacked = weights.pop("mlp.experts.gate_up_proj")
                    for part, begin in (("gate", 0), ("up", c.intermediate_size)):
                        weights[f"mlp.experts.{part}_proj"] = F.tensors.slice(stacked, starts=(begin, ),
                                                                              ends=(begin + c.intermediate_size, ),
                                                                              axes=(1, ),
                                                                              name=f"layer_{layer}_expert_{part}")
                    layer_id = F.builtin.scalar_const(tensor_type("int32", ()), local_layer,
                                                      name=f"layer_{layer}_state_id")
                    state = gdn if kind == "linear_attention" else paged
                    args = [hidden, state, layer_id]
                    if kind == "full_attention":
                        advance = F.builtin.scalar_const(tensor_type("bool", ()), indices[kind] == counts[kind],
                                                         name=f"layer_{layer}_advance_sequence")
                        args.extend((advance, cosine, sine))
                    args.extend(weights[key] for key in keys)
                    call = F.builtin.call(
                        *args, callee=name, result_type=TupleType((hidden.type, state.type)), effect=effect(
                            "read_write",
                            "gated_delta_net_state" if kind == "linear_attention" else "paged_attention_kv_cache"),
                        name=f"layer_{layer}_call", metadata={"layer_index": layer, "state_layer_index": local_layer})
                    hidden, updated = F.tensors.get_items(call, 0, 1, name_prefix=f"layer_{layer}")
                    if kind == "linear_attention":
                        gdn = updated
                    else:
                        paged = updated
                norm_weight = self.weight_at(importer.model_prefix + "norm.weight",
                                             tensor_type("bfloat16", (c.hidden_size, )))
                if importer.execution_phase == "prefill":
                    hidden = F.tensors.slice(hidden, starts=(importer.num_tokens - 1,), ends=(importer.num_tokens,),
                                             axes=(0,), name="prefill_last_hidden")
                hidden = rms_norm(hidden, norm_weight, c.epsilon, name="final_norm")
                lm_head = embedding if c.tie_word_embeddings else self.weight_at(
                    "lm_head.weight", tensor_type("bfloat16", (c.vocab_size, c.hidden_size)))
                logits = F.tensors.cast(linear(hidden, lm_head, name="lm_head"), "float32", name="logits")
                token = F.nn.greedy_sample(logits, name="next_token")
                self.function("main", (ids, entry_gdn, entry_paged), (logits, token, gdn, paged))

        metadata = {
            "architecture": "Qwen3_5MoeForConditionalGeneration",
            "model_type": "qwen3_5_moe",
            "mode": "decode-1" if self.execution_phase == "decode" else "prefill",
            "execution_phase": self.execution_phase,
            "tokens_per_call": self.num_tokens,
            "scope": "text-decoder",
            "revision": self.revision,
            "model_prefix": self.model_prefix,
            "num_hidden_layers": len(self.layers),
            "source_num_hidden_layers": c.num_hidden_layers,
            "imported_layer_indices": self.layers,
            "layer_types": tuple(c.layer_types[index] for index in self.layers),
            "hidden_size": c.hidden_size,
            "vocab_size": c.vocab_size,
            "num_experts": c.num_experts,
            "num_experts_per_tok": c.num_experts_per_tok,
            "numerical_contract": "nncase",
        }
        graph = Graph(dialect="high_level", stage="imported", entry="main", metadata=metadata).build()
        return verify_module(
            attach_import_source_locations(graph, architecture=metadata["architecture"], revision=self.revision))
