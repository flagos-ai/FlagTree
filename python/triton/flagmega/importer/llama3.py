# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Llama 3.x dense decoder importer.

Llama-3.2-1B is the first target: bias-free GQA attention without QK-norm,
SiLU/SwiGLU MLP, tied LM head and Llama 3 scaled RoPE (``rope_type:
llama3``).  The scaled-RoPE contract is carried as the exact FP32 inverse
frequency table so the semantic ``nn.rotary_embedding`` op does not have to
re-derive a relaxed frequency schedule from ``theta`` alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any, Mapping

from triton.flagmega.errors import ImporterError
from triton.flagmega.importer.checkpoint import Checkpoint, DirectoryCheckpoint, TensorInfo
from triton.flagmega.importer.qwen3 import (
    _build_attention_dataflow,
    _build_rotary_embedding,
    _optional_int,
    _positive_int,
    _validate_execution,
)
from triton.flagmega.importer.source import attach_import_source_locations
from triton.flagmega.ir import (
    DType,
    IRBuilder,
    IRModule,
    TupleType,
    effect,
    get_definition,
    tensor_type,
    verify_module,
)
from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionStateConfig


@dataclass(frozen=True)
class Llama3LayerConfig:
    vocab_size: int
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    epsilon: float
    rope_theta: float
    max_position_embeddings: int
    padding_idx: int | None
    tie_word_embeddings: bool
    rope_scaling: Mapping[str, Any] | None = None

    @property
    def query_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_key_value_heads * self.head_dim


def llama3_inverse_frequencies(config: Llama3LayerConfig) -> tuple[float, ...]:
    """Return the checkpoint's exact FP32 rotary frequency table.

    Llama 3.x replaces the plain ``theta ** (-2i/d)`` schedule with a
    piecewise rescaling around ``original_max_position_embeddings``.  The
    importer materializes the same table HuggingFace derives, computed in
    FP32 exactly as ``transformers`` does, and pins it on the rotary op.
    """
    import torch

    scaling = config.rope_scaling
    if not scaling:
        raise ImporterError("Llama 3 import requires an explicit scaled RoPE schedule.")
    rope_type = str(scaling.get("rope_type", scaling.get("type", "")))
    if rope_type != "llama3":
        raise ImporterError(f"Unsupported Llama RoPE scaling type {rope_type!r}.")
    factor = float(scaling["factor"])
    low_freq_factor = float(scaling["low_freq_factor"])
    high_freq_factor = float(scaling["high_freq_factor"])
    original_max = float(scaling["original_max_position_embeddings"])
    if min(factor, low_freq_factor, high_freq_factor, original_max) <= 0:
        raise ImporterError("Llama RoPE scaling parameters must be positive.")
    exponents = torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim
    # Every step below stays in FP32 tensors, exactly like transformers'
    # ``_compute_llama3_parameters``; widening to Python floats would move
    # boundary entries by one ULP.
    inverse = 1.0 / (config.rope_theta**exponents)
    wavelength = 2 * pi / inverse
    low_wavelen = original_max / low_freq_factor
    high_wavelen = original_max / high_freq_factor
    smooth = (original_max / wavelength - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * inverse / factor + smooth * inverse
    # Wavelengths below ``high_wavelen`` keep the base schedule, those above
    # ``low_wavelen`` are divided by the factor, and the band in between is
    # smoothly interpolated.
    band = torch.where(wavelength > low_wavelen, inverse / factor, smoothed)
    return tuple(torch.where(wavelength < high_wavelen, inverse, band).tolist())


def _parse_config(config: Mapping[str, Any]) -> Llama3LayerConfig:
    architectures = tuple(str(value) for value in config.get("architectures", ()))
    if config.get("model_type") != "llama" or "LlamaForCausalLM" not in architectures:
        raise ImporterError("Expected a LlamaForCausalLM checkpoint.")
    if bool(config.get("attention_bias", False)) or bool(config.get("mlp_bias", False)):
        raise ImporterError("Llama import requires bias-free attention and MLP projections.")
    if str(config.get("hidden_act", "silu")) not in {"silu", "swish"}:
        raise ImporterError("Llama import requires the SiLU activation.")
    num_heads = _positive_int(config, "num_attention_heads")
    head_dim = int(config.get("head_dim") or _positive_int(config, "hidden_size") // num_heads)
    result = Llama3LayerConfig(
        vocab_size=_positive_int(config, "vocab_size"),
        num_hidden_layers=_positive_int(config, "num_hidden_layers"),
        hidden_size=_positive_int(config, "hidden_size"),
        intermediate_size=_positive_int(config, "intermediate_size"),
        num_attention_heads=num_heads,
        num_key_value_heads=_positive_int(config, "num_key_value_heads"),
        head_dim=head_dim,
        epsilon=float(config.get("rms_norm_eps", 1e-6)),
        rope_theta=float(config.get("rope_theta", 10_000.0)),
        max_position_embeddings=_positive_int(config, "max_position_embeddings"),
        padding_idx=_optional_int(config.get("pad_token_id")),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        rope_scaling=dict(config["rope_scaling"]) if config.get("rope_scaling") else None,
    )
    if result.query_size != result.hidden_size:
        raise ImporterError("Llama import requires num_attention_heads * head_dim == hidden_size.")
    if result.num_attention_heads % result.num_key_value_heads:
        raise ImporterError("Llama attention heads must be divisible by KV heads.")
    if result.epsilon <= 0 or result.rope_theta <= 0:
        raise ImporterError("Llama RMS epsilon and RoPE theta must be positive.")
    return result


def _discover_layer_prefix(keys: tuple[str, ...], layer: int) -> str:
    suffix = f"layers.{layer}.input_layernorm.weight"
    matches = [key[:-len("input_layernorm.weight")] for key in keys if key.endswith(suffix)]
    if len(matches) != 1:
        raise ImporterError(f"Expected one Llama layer-{layer} namespace, got {matches}.")
    return matches[0]


def _require_tensor(checkpoint: Checkpoint, key: str, shape: tuple[int, ...], dtype: DType) -> TensorInfo:
    info = checkpoint.tensor_info(key)
    if info.shape != shape:
        raise ImporterError(f"Tensor {key!r} must have shape {shape}, got {info.shape}.")
    if info.dtype != dtype:
        raise ImporterError(f"Tensor {key!r} must use {dtype.value}, got {info.dtype.value}.")
    return info


class Llama3LayerImporter:
    """Import the first dense Llama decoder layer for one-token decode.

    The entry boundary intentionally matches the reference no-lm-head regression:
    token embedding, decoder layer, final RMSNorm and FP32 hidden output.
    """

    def __init__(
        self,
        checkpoint: Checkpoint | str,
        *,
        layer: int = 0,
        block_size: int = 256,
        num_blocks: int = 16,
        revision: str | None = None,
        execution_phase: str = "decode-1",
        num_tokens: int = 1,
    ) -> None:
        if layer != 0:
            raise ImporterError("Llama single-layer import currently imports exactly layer 0.")
        self.execution_phase = execution_phase
        self.num_tokens = _validate_execution(execution_phase, num_tokens)
        self.checkpoint = DirectoryCheckpoint(checkpoint) if isinstance(checkpoint, str) else checkpoint
        self.layer = layer
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.revision = revision
        self.config = _parse_config(self.checkpoint.config)
        self.inverse_frequencies = llama3_inverse_frequencies(self.config)
        self.layer_prefix = _discover_layer_prefix(self.checkpoint.keys, layer)
        self.model_prefix = self.layer_prefix[:-len(f"layers.{layer}.")]
        self.state_config = PagedAttentionStateConfig(
            self.config.num_hidden_layers,
            self.config.num_key_value_heads,
            self.config.head_dim,
            block_size=block_size,
            num_blocks=num_blocks,
        )

    def import_module(self) -> IRModule:
        config = self.config
        builder = IRBuilder(
            dialect="high_level",
            stage="imported",
            metadata={
                "architecture": "LlamaForCausalLM",
                "model_type": "llama",
                "layer": self.layer,
                "num_layers": 1,
                "mode": self.execution_phase,
                "tokens_per_call": self.num_tokens,
                "output_boundary": "final_norm_hidden_fp32",
                "revision": self.revision,
                "layer_prefix": self.layer_prefix,
                "model_prefix": self.model_prefix,
                "vocab_size": config.vocab_size,
                "padding_idx": config.padding_idx,
                "num_hidden_layers": config.num_hidden_layers,
                "hidden_size": config.hidden_size,
                "intermediate_size": config.intermediate_size,
                "num_attention_heads": config.num_attention_heads,
                "num_key_value_heads": config.num_key_value_heads,
                "head_dim": config.head_dim,
                "rope_theta": config.rope_theta,
                "rope_scaling": dict(config.rope_scaling or {}),
                "max_position_embeddings": config.max_position_embeddings,
                "paged_attention": {
                    "block_size": self.block_size,
                    "num_blocks": self.num_blocks,
                    "cache_layout": [
                        "NumBlocks",
                        "NumLayers",
                        "KV",
                        "BlockSize",
                        "NumKVHeads",
                        "HeadDim",
                    ],
                    "vectorized_axes": ["HeadDim"],
                    "lanes": [8],
                },
            },
        )
        tokens = self.num_tokens
        hidden_type = tensor_type(DType.BFLOAT16, [tokens, config.hidden_size])
        intermediate_type = tensor_type(DType.BFLOAT16, [tokens, config.intermediate_size])
        output_type = tensor_type(DType.FLOAT32, [tokens, config.hidden_size])
        state_type = self.state_config.ref_type
        input_ids = builder.var("input_ids", tensor_type(DType.INT32, [tokens]), id="input_ids")
        state = builder.var("paged_attention_kv_cache", state_type, id="paged_attention_kv_cache")

        def weight_at(name: str, key: str, shape: tuple[int, ...]):
            info = _require_tensor(self.checkpoint, key, shape, DType.BFLOAT16)
            return builder.weight(name, tensor_type(info.dtype, info.shape), source=info.source, key=key,
                                  id="w_" + name.replace(".", "_"))

        def weight(name: str, shape: tuple[int, ...]):
            return weight_at(name, self.layer_prefix + name, shape)

        embedding_weight = weight_at(
            "embed_tokens.weight",
            self.model_prefix + "embed_tokens.weight",
            (config.vocab_size, config.hidden_size),
        )
        hidden = builder.call("nn.embedding", [input_ids, embedding_weight], hidden_type, id="token_embedding",
                              attrs={"padding_idx": config.padding_idx})
        input_norm_weight = weight("input_layernorm.weight", (config.hidden_size, ))
        normalized = builder.call("nn.rms_norm", [hidden, input_norm_weight], hidden_type, id="input_norm",
                                  attrs={"epsilon": config.epsilon, "weight_bias": 0.0})

        q_weight = weight("self_attn.q_proj.weight", (config.query_size, config.hidden_size))
        k_weight = weight("self_attn.k_proj.weight", (config.kv_size, config.hidden_size))
        v_weight = weight("self_attn.v_proj.weight", (config.kv_size, config.hidden_size))
        o_weight = weight("self_attn.o_proj.weight", (config.hidden_size, config.query_size))
        layer_id = builder.call("builtin.scalar_const", [], tensor_type(DType.INT32, []), id="layer_id",
                                attrs={"value": self.layer})
        advance_sequence = builder.call("builtin.scalar_const", [], tensor_type(DType.BOOL, []), id="advance_sequence",
                                        attrs={"value": True})
        attention_output, updated_state = _build_attention_dataflow(
            builder,
            prefix="self_attention",
            value=normalized,
            state=state,
            q_weight=q_weight,
            k_weight=k_weight,
            v_weight=v_weight,
            q_norm_weight=None,
            k_norm_weight=None,
            output_weight=o_weight,
            layer_id=layer_id,
            advance_sequence=advance_sequence,
            config=config,
            output_id="attention_output",
            updated_state_id="updated_state",
            inverse_frequencies=self.inverse_frequencies,
        )
        after_attention = builder.call("math.add", [hidden, attention_output], hidden_type, id="after_attention")

        post_norm_weight = weight("post_attention_layernorm.weight", (config.hidden_size, ))
        mlp_input = builder.call("nn.rms_norm", [after_attention, post_norm_weight], hidden_type,
                                 id="post_attention_norm", attrs={"epsilon": config.epsilon, "weight_bias": 0.0})
        gate_weight = weight("mlp.gate_proj.weight", (config.intermediate_size, config.hidden_size))
        up_weight = weight("mlp.up_proj.weight", (config.intermediate_size, config.hidden_size))
        mlp_gate_up = builder.call("nn.dense_matmul_glu", [mlp_input, gate_weight, up_weight], intermediate_type,
                                   id="mlp_gate_up",
                                   attrs=get_definition("nn.dense_matmul_glu").normalize_attrs({"activation": "silu"}))
        down_weight = weight("mlp.down_proj.weight", (config.hidden_size, config.intermediate_size))
        mlp_output = builder.call("math.matmul", [mlp_gate_up, down_weight], hidden_type, id="mlp_down",
                                  attrs={"transpose_a": False, "transpose_b": True})
        decoder_output = builder.call("math.add", [after_attention, mlp_output], hidden_type, id="decoder_output")

        final_norm_weight = weight_at("norm.weight", self.model_prefix + "norm.weight", (config.hidden_size, ))
        final_hidden = builder.call("nn.rms_norm", [decoder_output, final_norm_weight], hidden_type, id="final_norm",
                                    attrs={"epsilon": config.epsilon, "weight_bias": 0.0})
        output = builder.call("tensors.cast", [final_hidden], output_type, id="output",
                              attrs={"dtype": DType.FLOAT32.value})
        builder.function("main", [input_ids, state], [output, updated_state])
        return verify_module(
            attach_import_source_locations(
                builder.build(entry="main"),
                architecture="LlamaForCausalLM",
                revision=self.revision,
            ))


class Llama3ModelImporter:
    """Import a complete dense Llama causal-LM decode step.

    ``main`` owns embedding, per-layer weights, LM head and sampling while
    every layer calls one reusable semantic ``decode_layer`` function, the
    same boundary contract the Qwen3 importer establishes.
    """

    def __init__(
        self,
        checkpoint: Checkpoint | str,
        *,
        block_size: int = 256,
        num_blocks: int = 16,
        revision: str | None = None,
        execution_phase: str = "decode-1",
        num_tokens: int = 1,
    ) -> None:
        self.execution_phase = execution_phase
        self.num_tokens = _validate_execution(execution_phase, num_tokens)
        self.checkpoint = (DirectoryCheckpoint(checkpoint) if isinstance(checkpoint, str) else checkpoint)
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.revision = revision
        self.config = _parse_config(self.checkpoint.config)
        self.inverse_frequencies = llama3_inverse_frequencies(self.config)
        self.layer_prefixes = tuple(
            _discover_layer_prefix(self.checkpoint.keys, layer) for layer in range(self.config.num_hidden_layers))
        model_prefixes = {prefix[:-len(f"layers.{layer}.")] for layer, prefix in enumerate(self.layer_prefixes)}
        if len(model_prefixes) != 1:
            raise ImporterError(f"Llama decoder layers do not share one model namespace: {sorted(model_prefixes)}.")
        self.model_prefix = next(iter(model_prefixes))
        self.state_config = PagedAttentionStateConfig(
            self.config.num_hidden_layers,
            self.config.num_key_value_heads,
            self.config.head_dim,
            block_size=block_size,
            num_blocks=num_blocks,
        )

    def import_module(self) -> IRModule:
        config = self.config
        builder = IRBuilder(
            dialect="high_level",
            stage="imported",
            metadata={
                "architecture": "LlamaForCausalLM",
                "model_type": "llama",
                "layer": None,
                "num_layers": config.num_hidden_layers,
                "mode": self.execution_phase,
                "tokens_per_call": self.num_tokens,
                "output_boundary":
                ("logits_fp32" if self.execution_phase == "prefill" else "logits_fp32_and_greedy_token"),
                "revision": self.revision,
                "layer_prefixes": self.layer_prefixes,
                "model_prefix": self.model_prefix,
                "vocab_size": config.vocab_size,
                "padding_idx": config.padding_idx,
                "num_hidden_layers": config.num_hidden_layers,
                "hidden_size": config.hidden_size,
                "intermediate_size": config.intermediate_size,
                "num_attention_heads": config.num_attention_heads,
                "num_key_value_heads": config.num_key_value_heads,
                "head_dim": config.head_dim,
                "rope_theta": config.rope_theta,
                "rope_scaling": dict(config.rope_scaling or {}),
                "max_position_embeddings": config.max_position_embeddings,
                "tie_word_embeddings": config.tie_word_embeddings,
                "paged_attention": {
                    "block_size": self.block_size,
                    "num_blocks": self.num_blocks,
                    "cache_layout": [
                        "NumBlocks",
                        "NumLayers",
                        "KV",
                        "BlockSize",
                        "NumKVHeads",
                        "HeadDim",
                    ],
                    "vectorized_axes": ["HeadDim"],
                    "lanes": [8],
                },
            },
        )
        tokens = self.num_tokens
        hidden_type = tensor_type(DType.BFLOAT16, [tokens, config.hidden_size])
        intermediate_type = tensor_type(DType.BFLOAT16, [tokens, config.intermediate_size])
        logits_bf16_type = tensor_type(DType.BFLOAT16, [tokens, config.vocab_size])
        logits_type = tensor_type(DType.FLOAT32, [tokens, config.vocab_size])
        token_type = tensor_type(DType.INT32, [1])
        state_type = self.state_config.ref_type
        input_ids = builder.var("input_ids", tensor_type(DType.INT32, [tokens]), id="input_ids")
        state = builder.var("paged_attention_kv_cache", state_type, id="paged_attention_kv_cache")
        entry_state = state

        def weight_at(
            name: str,
            key: str,
            shape: tuple[int, ...],
            *,
            node_id: str,
            rdata_group: Mapping[str, object] | None = None,
        ):
            info = _require_tensor(self.checkpoint, key, shape, DType.BFLOAT16)
            return builder.weight(
                name,
                tensor_type(info.dtype, info.shape),
                source=info.source,
                key=key,
                id=node_id,
                metadata=({} if rdata_group is None else {"rdata_group": dict(rdata_group)}),
            )

        embedding_weight = weight_at(
            "embed_tokens.weight",
            self.model_prefix + "embed_tokens.weight",
            (config.vocab_size, config.hidden_size),
            node_id="w_embed_tokens_weight",
        )
        hidden = builder.call(
            "nn.embedding",
            [input_ids, embedding_weight],
            hidden_type,
            id="token_embedding",
            attrs={"padding_idx": config.padding_idx},
        )

        def decode_parameter(name: str, value_type):
            return builder.var(
                name,
                value_type,
                id=f"decode_layer_{name}",
                metadata={"function_parameter": "decode_layer"},
            )

        # Positions belong to the token step and are shared across layers;
        # cache writes do not advance the step until the final decoder call.
        position_embeddings = _build_rotary_embedding(
            builder,
            prefix="main",
            value=hidden,
            state=state,
            config=config,
            inverse_frequencies=self.inverse_frequencies,
        )
        decode_cos = decode_parameter("rotary_cos", position_embeddings[0].type)
        decode_sin = decode_parameter("rotary_sin", position_embeddings[1].type)

        decode_hidden = decode_parameter("hidden", hidden_type)
        decode_state = decode_parameter("state", state_type)
        decode_input_norm_weight = decode_parameter("input_norm_weight",
                                                    tensor_type(DType.BFLOAT16, [config.hidden_size]))
        decode_q_weight = decode_parameter("q_weight",
                                           tensor_type(DType.BFLOAT16, [config.query_size, config.hidden_size]))
        decode_k_weight = decode_parameter("k_weight", tensor_type(DType.BFLOAT16,
                                                                   [config.kv_size, config.hidden_size]))
        decode_v_weight = decode_parameter("v_weight", tensor_type(DType.BFLOAT16,
                                                                   [config.kv_size, config.hidden_size]))
        decode_o_weight = decode_parameter("o_weight",
                                           tensor_type(DType.BFLOAT16, [config.hidden_size, config.query_size]))
        decode_post_norm_weight = decode_parameter("post_norm_weight", tensor_type(DType.BFLOAT16,
                                                                                   [config.hidden_size]))
        decode_gate_weight = decode_parameter(
            "gate_weight",
            tensor_type(DType.BFLOAT16, [config.intermediate_size, config.hidden_size]),
        )
        decode_up_weight = decode_parameter(
            "up_weight",
            tensor_type(DType.BFLOAT16, [config.intermediate_size, config.hidden_size]),
        )
        decode_down_weight = decode_parameter(
            "down_weight",
            tensor_type(DType.BFLOAT16, [config.hidden_size, config.intermediate_size]),
        )
        decode_layer_id = decode_parameter("layer_id", tensor_type(DType.INT32, []))
        decode_advance_sequence = decode_parameter("advance_sequence", tensor_type(DType.BOOL, []))
        decode_parameters = (
            decode_hidden,
            decode_state,
            decode_input_norm_weight,
            decode_q_weight,
            decode_k_weight,
            decode_v_weight,
            decode_o_weight,
            decode_post_norm_weight,
            decode_gate_weight,
            decode_up_weight,
            decode_down_weight,
            decode_layer_id,
            decode_advance_sequence,
            decode_cos,
            decode_sin,
        )

        decode_normalized = builder.call(
            "nn.rms_norm",
            [decode_hidden, decode_input_norm_weight],
            hidden_type,
            id="decode_layer_input_norm",
            attrs={"epsilon": config.epsilon, "weight_bias": 0.0},
        )
        decode_attention_output, decode_updated_state = _build_attention_dataflow(
            builder,
            prefix="decode_layer",
            value=decode_normalized,
            state=decode_state,
            q_weight=decode_q_weight,
            k_weight=decode_k_weight,
            v_weight=decode_v_weight,
            q_norm_weight=None,
            k_norm_weight=None,
            output_weight=decode_o_weight,
            layer_id=decode_layer_id,
            advance_sequence=decode_advance_sequence,
            config=config,
            output_id="decode_layer_attention_output",
            updated_state_id="decode_layer_updated_state",
            position_embeddings=(decode_cos, decode_sin),
        )
        decode_after_attention = builder.call(
            "math.add",
            [decode_hidden, decode_attention_output],
            hidden_type,
            id="decode_layer_after_attention",
        )
        decode_mlp_input = builder.call(
            "nn.rms_norm",
            [decode_after_attention, decode_post_norm_weight],
            hidden_type,
            id="decode_layer_post_attention_norm",
            attrs={"epsilon": config.epsilon, "weight_bias": 0.0},
        )
        decode_mlp_gate_up = builder.call(
            "nn.dense_matmul_glu",
            [decode_mlp_input, decode_gate_weight, decode_up_weight],
            intermediate_type,
            id="decode_layer_mlp_gate_up",
            attrs=get_definition("nn.dense_matmul_glu").normalize_attrs({"activation": "silu"}),
        )
        decode_mlp_output = builder.call(
            "math.matmul",
            [decode_mlp_gate_up, decode_down_weight],
            hidden_type,
            id="decode_layer_mlp_down",
            attrs={"transpose_a": False, "transpose_b": True},
        )
        decode_output = builder.call(
            "math.add",
            [decode_after_attention, decode_mlp_output],
            hidden_type,
            id="decode_layer_output",
        )

        for layer, layer_prefix in enumerate(self.layer_prefixes):
            prefix = f"layer_{layer}"

            def layer_weight(name: str, shape: tuple[int, ...], layer: int = layer, layer_prefix: str = layer_prefix,
                             prefix: str = prefix):
                return weight_at(
                    f"layers.{layer}.{name}",
                    layer_prefix + name,
                    shape,
                    node_id=f"w_{prefix}_{name.replace('.', '_')}",
                    rdata_group={
                        "name": f"llama3.layer.{name}",
                        "index": layer,
                        "count": config.num_hidden_layers,
                    },
                )

            input_norm_weight = layer_weight("input_layernorm.weight", (config.hidden_size, ))
            q_weight = layer_weight("self_attn.q_proj.weight", (config.query_size, config.hidden_size))
            k_weight = layer_weight("self_attn.k_proj.weight", (config.kv_size, config.hidden_size))
            v_weight = layer_weight("self_attn.v_proj.weight", (config.kv_size, config.hidden_size))
            o_weight = layer_weight("self_attn.o_proj.weight", (config.hidden_size, config.query_size))
            post_norm_weight = layer_weight("post_attention_layernorm.weight", (config.hidden_size, ))
            gate_weight = layer_weight("mlp.gate_proj.weight", (config.intermediate_size, config.hidden_size))
            up_weight = layer_weight("mlp.up_proj.weight", (config.intermediate_size, config.hidden_size))
            down_weight = layer_weight("mlp.down_proj.weight", (config.hidden_size, config.intermediate_size))
            layer_id = builder.call(
                "builtin.scalar_const",
                [],
                tensor_type(DType.INT32, []),
                id=f"{prefix}_id",
                attrs={"value": layer},
            )
            advance_sequence = builder.call(
                "builtin.scalar_const",
                [],
                tensor_type(DType.BOOL, []),
                id=f"{prefix}_advance_sequence",
                attrs={"value": layer + 1 == config.num_hidden_layers},
            )
            call = builder.call(
                "builtin.call",
                [
                    hidden,
                    state,
                    input_norm_weight,
                    q_weight,
                    k_weight,
                    v_weight,
                    o_weight,
                    post_norm_weight,
                    gate_weight,
                    up_weight,
                    down_weight,
                    layer_id,
                    advance_sequence,
                    *position_embeddings,
                ],
                TupleType((hidden_type, state_type)),
                id=f"{prefix}_decode_layer_call",
                effect=effect("read_write", "paged_attention_kv_cache"),
                attrs={"callee": "decode_layer"},
                metadata={"layer_index": layer},
            )
            hidden = builder.call(
                "builtin.get_item",
                [call],
                hidden_type,
                id=f"{prefix}_decoder_output",
                attrs={"index": 0},
            )
            state = builder.call(
                "builtin.get_item",
                [call],
                state_type,
                id=("updated_state" if layer + 1 == config.num_hidden_layers else f"{prefix}_updated_state"),
                attrs={"index": 1},
            )

        final_norm_weight = weight_at(
            "norm.weight",
            self.model_prefix + "norm.weight",
            (config.hidden_size, ),
            node_id="w_norm_weight",
        )
        head_input = hidden
        head_hidden_type = hidden_type
        head_logits_bf16_type = logits_bf16_type
        head_logits_type = logits_type
        if self.execution_phase == "prefill" and tokens > 1:
            # Only the last token's logits drive next-token prediction.
            head_hidden_type = tensor_type(DType.BFLOAT16, [1, config.hidden_size])
            head_logits_bf16_type = tensor_type(DType.BFLOAT16, [1, config.vocab_size])
            head_logits_type = tensor_type(DType.FLOAT32, [1, config.vocab_size])
            head_input = builder.call(
                "tensors.slice",
                [hidden],
                head_hidden_type,
                id="prefill_last_hidden",
                attrs={"starts": (tokens - 1, ), "ends": (tokens, ), "axes": (0, ), "steps": (1, )},
            )
        final_hidden = builder.call(
            "nn.rms_norm",
            [head_input, final_norm_weight],
            head_hidden_type,
            id="final_norm",
            attrs={"epsilon": config.epsilon, "weight_bias": 0.0},
        )
        if config.tie_word_embeddings:
            lm_head_weight = embedding_weight
        else:
            lm_head_weight = weight_at(
                "lm_head.weight",
                "lm_head.weight",
                (config.vocab_size, config.hidden_size),
                node_id="w_lm_head_weight",
            )
        logits_bf16 = builder.call(
            "math.matmul",
            [final_hidden, lm_head_weight],
            head_logits_bf16_type,
            id="lm_head",
            attrs={"transpose_a": False, "transpose_b": True},
        )
        logits = builder.call(
            "tensors.cast",
            [logits_bf16],
            head_logits_type,
            id="logits",
            attrs={"dtype": DType.FLOAT32.value},
        )
        if self.execution_phase == "prefill":
            builder.function("main", [input_ids, entry_state], [logits, state])
        else:
            next_token = builder.call("nn.greedy_sample", [logits], token_type, id="next_token")
            builder.function("main", [input_ids, entry_state], [logits, next_token, state])
        builder.function(
            "decode_layer",
            decode_parameters,
            [decode_output, decode_updated_state],
            attrs={
                "calling_convention": "device",
                "noinline": True,
                "reusable": True,
            },
        )
        return verify_module(
            attach_import_source_locations(
                builder.build(entry="main"),
                architecture="LlamaForCausalLM",
                revision=self.revision,
            ))


def import_llama3_layer(
    checkpoint: Checkpoint | str,
    *,
    layer: int = 0,
    block_size: int = 256,
    num_blocks: int = 16,
    revision: str | None = None,
) -> IRModule:
    return Llama3LayerImporter(checkpoint, layer=layer, block_size=block_size, num_blocks=num_blocks,
                               revision=revision).import_module()


def import_llama3_model(
    checkpoint: Checkpoint | str,
    *,
    block_size: int = 256,
    num_blocks: int = 16,
    revision: str | None = None,
    execution_phase: str = "decode-1",
    num_tokens: int = 1,
) -> IRModule:
    return Llama3ModelImporter(
        checkpoint,
        block_size=block_size,
        num_blocks=num_blocks,
        revision=revision,
        execution_phase=execution_phase,
        num_tokens=num_tokens,
    ).import_module()


__all__ = [
    "Llama3LayerConfig",
    "Llama3LayerImporter",
    "Llama3ModelImporter",
    "import_llama3_layer",
    "import_llama3_model",
    "llama3_inverse_frequencies",
]
