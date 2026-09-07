# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Qwen3 dense decoder-layer importer aligned with nncase's Qwen3 path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from triton.flagmega.errors import ImporterError
from triton.flagmega.importer.checkpoint import Checkpoint, DirectoryCheckpoint, TensorInfo
from triton.flagmega.importer.source import attach_import_source_locations
from triton.flagmega.ir import (
    DType,
    IRBuilder,
    IRModule,
    NoneType,
    TupleType,
    effect,
    tensor_type,
    verify_module,
)
from triton.flagmega.ir.ops.nn._paged_attention_state import (
    PagedAttentionStateConfig,
    paged_attention_state_config_from_type,
)
from triton.flagmega.ir.ops.tensors.pack import Pack
from triton.flagmega.ir.ops.tensors.unpack import Unpack


@dataclass(frozen=True)
class Qwen3LayerConfig:
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

    @property
    def query_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_key_value_heads * self.head_dim


def _build_rotary_embedding(builder, *, prefix, value, state, config):
    """Token-step position embeddings shared by all decoder invocations."""
    def node_id(suffix):
        return f"{prefix}_{suffix}" if prefix else suffix

    rotary_type = tensor_type(DType.FLOAT32, (1, 1, config.head_dim))
    rotary = builder.call(
        "nn.rotary_embedding",
        [value, state],
        TupleType((rotary_type, rotary_type)),
        id=node_id("rotary_embedding"),
        effect=effect("read", "paged_attention_kv_cache"),
        attrs={
            "head_dim": config.head_dim,
            "theta": config.rope_theta,
            "attention_scaling": 1.0,
        },
    )
    cosine = builder.call(
        "builtin.get_item",
        [rotary],
        rotary_type,
        id=node_id("rotary_cos"),
        attrs={"index": 0},
    )
    sine = builder.call(
        "builtin.get_item",
        [rotary],
        rotary_type,
        id=node_id("rotary_sin"),
        attrs={"index": 1},
    )
    return cosine, sine


def _build_attention_dataflow(
    builder: IRBuilder,
    *,
    prefix: str,
    value,
    state,
    q_weight,
    k_weight,
    v_weight,
    q_norm_weight,
    k_norm_weight,
    output_weight,
    layer_id,
    advance_sequence,
    config: Qwen3LayerConfig,
    output_id: str | None = None,
    updated_state_id: str | None = None,
    position_embeddings=None,
):
    """Build nncase's QKV/RoPE/cache/PagedAttention/O-projection boundaries."""

    def node_id(suffix: str) -> str:
        return f"{prefix}_{suffix}" if prefix else suffix

    def transpose_weight(role: str, weight, output_size: int):
        return builder.call(
            "tensors.permute",
            [weight],
            tensor_type(DType.BFLOAT16, (config.hidden_size, output_size)),
            id=node_id(f"{role}_weight_kn"),
            attrs={"axes": (1, 0)},
        )

    q_weight_kn = transpose_weight("q", q_weight, config.query_size)
    k_weight_kn = transpose_weight("k", k_weight, config.kv_size)
    v_weight_kn = transpose_weight("v", v_weight, config.kv_size)
    none = builder.call("builtin.none", [], NoneType(), id=node_id("qkv_none"))
    projected_type = TupleType((
        tensor_type(DType.BFLOAT16, (1, config.query_size)),
        tensor_type(DType.BFLOAT16, (1, config.kv_size)),
        tensor_type(DType.BFLOAT16, (1, config.kv_size)),
    ))
    projected = builder.call(
        "nn.qkv_parallel_linear",
        [
            value,
            q_weight_kn,
            k_weight_kn,
            v_weight_kn,
            none,
            none,
            none,
            none,
            none,
            none,
            none,
            none,
            none,
        ],
        projected_type,
        id=node_id("qkv_projection"),
        attrs={
            "num_heads": config.num_attention_heads,
            "num_kv_heads": config.num_key_value_heads,
            "output_data_type": DType.BFLOAT16.value,
        },
    )

    def projection(role: str, index: int, heads: int):
        flat_type = projected_type.fields[index]
        flat = builder.call(
            "builtin.get_item",
            [projected],
            flat_type,
            id=node_id(f"{role}_flat"),
            attrs={"index": index},
        )
        return builder.call(
            "tensors.reshape",
            [flat],
            tensor_type(DType.BFLOAT16, (1, heads, config.head_dim)),
            id=node_id(role),
            attrs={"shape": (1, heads, config.head_dim)},
        )

    query = projection("query", 0, config.num_attention_heads)
    key = projection("key", 1, config.num_key_value_heads)
    value_slots = projection("value", 2, config.num_key_value_heads)
    query = builder.call(
        "nn.rms_norm",
        [query, q_norm_weight],
        query.type,
        id=node_id("query_norm"),
        attrs={"epsilon": config.epsilon, "weight_bias": 0.0},
    )
    key = builder.call(
        "nn.rms_norm",
        [key, k_norm_weight],
        key.type,
        id=node_id("key_norm"),
        attrs={"epsilon": config.epsilon, "weight_bias": 0.0},
    )
    cosine, sine = (
        _build_rotary_embedding(builder, prefix=prefix, value=value, state=state, config=config)
        if position_embeddings is None else position_embeddings
    )
    query = builder.call(
        "nn.rope", [query, cosine, sine], query.type, id=node_id("query_rope"))
    key = builder.call(
        "nn.rope", [key, cosine, sine], key.type, id=node_id("key_rope"))

    cache_config = paged_attention_state_config_from_type(state.type)
    attention_layout = ("seq", "head", "dim")
    cache_pack_attrs = {
        "lanes": (cache_config.lanes,),
        "axes": (attention_layout.index("dim"),),
    }

    def cache_pack(role: str, tensor):
        packed_type = Pack.infer_type((tensor,), cache_pack_attrs)
        return builder.call(
            "tensors.pack",
            [tensor],
            packed_type,
            id=node_id(f"{role}_cache_pack"),
            attrs=cache_pack_attrs,
        )

    packed_query = cache_pack("query", query)
    packed_key = cache_pack("key", key)
    packed_value = cache_pack("value", value_slots)
    do_not_advance = builder.call(
        "builtin.scalar_const",
        [],
        tensor_type(DType.BOOL, []),
        id=node_id("key_do_not_advance"),
        attrs={"value": False},
    )
    key_state = builder.call(
        "nn.update_paged_attention_kv_cache",
        [packed_key, state, layer_id, do_not_advance],
        state.type,
        id=node_id("key_cache_update"),
        effect=effect("read_write", "paged_attention_kv_cache"),
        attrs={"cache_kind": "key", "layout": attention_layout},
    )
    updated_state = builder.call(
        "nn.update_paged_attention_kv_cache",
        [packed_value, key_state, layer_id, advance_sequence],
        state.type,
        id=updated_state_id or node_id("value_cache_update"),
        effect=effect("read_write", "paged_attention_kv_cache"),
        attrs={"cache_kind": "value", "layout": attention_layout},
    )
    attended = builder.call(
        "nn.paged_attention",
        [packed_query, updated_state, layer_id],
        packed_query.type,
        id=node_id("paged_attention"),
        effect=effect("read", "paged_attention_kv_cache"),
        attrs={
            "scale": config.head_dim ** -0.5,
            "layout": attention_layout,
            "hidden_size": config.query_size,
        },
    )
    unpack_attrs = {"axes": (attention_layout.index("dim"),)}
    attended = builder.call(
        "tensors.unpack",
        [attended],
        Unpack.infer_type((attended,), unpack_attrs),
        id=node_id("paged_attention_unpacked"),
        attrs=unpack_attrs,
    )
    merged = builder.call(
        "tensors.reshape",
        [attended],
        tensor_type(DType.BFLOAT16, (1, config.query_size)),
        id=node_id("attention_merged"),
        attrs={"shape": (1, config.query_size)},
    )
    output = builder.call(
        "math.matmul",
        [merged, output_weight],
        tensor_type(DType.BFLOAT16, (1, config.hidden_size)),
        id=output_id or node_id("attention_output"),
        attrs={"transpose_a": False, "transpose_b": True},
    )
    return output, updated_state


class Qwen3LayerImporter:
    """Import the first dense Qwen3 decoder layer for one-token decode.

    The entry boundary intentionally matches the nncase no-lm-head regression:
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
    ) -> None:
        if layer != 0:
            raise ImporterError("Qwen3 single-layer P0 currently imports exactly layer 0.")
        self.checkpoint = DirectoryCheckpoint(checkpoint) if isinstance(checkpoint, str) else checkpoint
        self.layer = layer
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.revision = revision
        self.config = self._parse_config(self.checkpoint.config)
        self.layer_prefix = self._discover_layer_prefix(self.checkpoint.keys, layer)
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
                "architecture": "Qwen3ForCausalLM",
                "model_type": "qwen3",
                "layer": self.layer,
                "num_layers": 1,
                "mode": "decode-1",
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
                "max_position_embeddings": config.max_position_embeddings,
                "paged_attention": {
                    "block_size": self.block_size,
                    "num_blocks": self.num_blocks,
                    "cache_layout": [
                        "NumBlocks", "NumLayers", "KV", "BlockSize", "NumKVHeads", "HeadDim",
                    ],
                    "vectorized_axes": ["HeadDim"],
                    "lanes": [8],
                },
            },
        )
        hidden_type = tensor_type(DType.BFLOAT16, [1, config.hidden_size])
        intermediate_type = tensor_type(DType.BFLOAT16, [1, config.intermediate_size])
        output_type = tensor_type(DType.FLOAT32, [1, config.hidden_size])
        state_type = self.state_config.ref_type
        input_ids = builder.var("input_ids", tensor_type(DType.INT32, [1]), id="input_ids")
        state = builder.var("paged_attention_kv_cache", state_type, id="paged_attention_kv_cache")

        def weight_at(name: str, key: str, shape: tuple[int, ...]):
            info = self._require_tensor(key, shape, DType.BFLOAT16)
            return builder.weight(
                name, tensor_type(info.dtype, info.shape), source=info.source, key=key,
                id="w_" + name.replace(".", "_"))

        def weight(name: str, shape: tuple[int, ...]):
            return weight_at(name, self.layer_prefix + name, shape)

        embedding_weight = weight_at(
            "embed_tokens.weight",
            self.model_prefix + "embed_tokens.weight",
            (config.vocab_size, config.hidden_size),
        )
        hidden = builder.call(
            "nn.embedding", [input_ids, embedding_weight], hidden_type,
            id="token_embedding", attrs={"padding_idx": config.padding_idx})
        input_norm_weight = weight("input_layernorm.weight", (config.hidden_size, ))
        normalized = builder.call(
            "nn.rms_norm", [hidden, input_norm_weight], hidden_type,
            id="input_norm", attrs={"epsilon": config.epsilon, "weight_bias": 0.0})

        q_weight = weight("self_attn.q_proj.weight", (config.query_size, config.hidden_size))
        k_weight = weight("self_attn.k_proj.weight", (config.kv_size, config.hidden_size))
        v_weight = weight("self_attn.v_proj.weight", (config.kv_size, config.hidden_size))
        q_norm_weight = weight("self_attn.q_norm.weight", (config.head_dim, ))
        k_norm_weight = weight("self_attn.k_norm.weight", (config.head_dim, ))
        o_weight = weight("self_attn.o_proj.weight", (config.hidden_size, config.query_size))
        layer_id = builder.call(
            "builtin.scalar_const", [], tensor_type(DType.INT32, []),
            id="layer_id", attrs={"value": self.layer})
        advance_sequence = builder.call(
            "builtin.scalar_const", [], tensor_type(DType.BOOL, []),
            id="advance_sequence", attrs={"value": True})
        attention_output, updated_state = _build_attention_dataflow(
            builder,
            prefix="self_attention",
            value=normalized,
            state=state,
            q_weight=q_weight,
            k_weight=k_weight,
            v_weight=v_weight,
            q_norm_weight=q_norm_weight,
            k_norm_weight=k_norm_weight,
            output_weight=o_weight,
            layer_id=layer_id,
            advance_sequence=advance_sequence,
            config=config,
            output_id="attention_output",
            updated_state_id="updated_state",
        )
        after_attention = builder.call(
            "math.add", [hidden, attention_output], hidden_type, id="after_attention")

        post_norm_weight = weight("post_attention_layernorm.weight", (config.hidden_size, ))
        mlp_input = builder.call(
            "nn.rms_norm", [after_attention, post_norm_weight], hidden_type,
            id="post_attention_norm", attrs={"epsilon": config.epsilon, "weight_bias": 0.0})
        gate_weight = weight("mlp.gate_proj.weight", (config.intermediate_size, config.hidden_size))
        up_weight = weight("mlp.up_proj.weight", (config.intermediate_size, config.hidden_size))
        mlp_gate_up = builder.call(
            "nn.dense_matmul_glu", [mlp_input, gate_weight, up_weight], intermediate_type,
            id="mlp_gate_up", attrs={"activation": "silu"})
        down_weight = weight("mlp.down_proj.weight", (config.hidden_size, config.intermediate_size))
        mlp_output = builder.call(
            "math.matmul", [mlp_gate_up, down_weight], hidden_type,
            id="mlp_down", attrs={"transpose_a": False, "transpose_b": True})
        decoder_output = builder.call(
            "math.add", [after_attention, mlp_output], hidden_type, id="decoder_output")

        final_norm_weight = weight_at(
            "norm.weight", self.model_prefix + "norm.weight", (config.hidden_size, ))
        final_hidden = builder.call(
            "nn.rms_norm", [decoder_output, final_norm_weight], hidden_type,
            id="final_norm", attrs={"epsilon": config.epsilon, "weight_bias": 0.0})
        output = builder.call(
            "tensors.cast", [final_hidden], output_type, id="output",
            attrs={"dtype": DType.FLOAT32.value})
        builder.function("main", [input_ids, state], [output, updated_state])
        return verify_module(attach_import_source_locations(
            builder.build(entry="main"),
            architecture="Qwen3ForCausalLM",
            revision=self.revision,
        ))

    def _require_tensor(self, key: str, shape: tuple[int, ...], dtype: DType) -> TensorInfo:
        info = self.checkpoint.tensor_info(key)
        if info.shape != shape:
            raise ImporterError(f"Tensor {key!r} must have shape {shape}, got {info.shape}.")
        if info.dtype != dtype:
            raise ImporterError(f"Tensor {key!r} must use {dtype.value}, got {info.dtype.value}.")
        return info

    @staticmethod
    def _discover_layer_prefix(keys: tuple[str, ...], layer: int) -> str:
        suffix = f"layers.{layer}.input_layernorm.weight"
        matches = [key[:-len("input_layernorm.weight")] for key in keys if key.endswith(suffix)]
        if len(matches) != 1:
            raise ImporterError(f"Expected one Qwen3 layer-{layer} namespace, got {matches}.")
        return matches[0]

    @staticmethod
    def _parse_config(config: Mapping[str, Any]) -> Qwen3LayerConfig:
        architectures = tuple(str(value) for value in config.get("architectures", ()))
        if config.get("model_type") != "qwen3" or "Qwen3ForCausalLM" not in architectures:
            raise ImporterError("Expected a Qwen3ForCausalLM checkpoint.")
        if bool(config.get("attention_bias", False)) or bool(config.get("mlp_bias", False)):
            raise ImporterError("Qwen3 single-layer P0 requires bias-free attention and MLP projections.")
        if str(config.get("hidden_act", "silu")) not in {"silu", "swish"}:
            raise ImporterError("Qwen3 single-layer P0 requires the SiLU activation.")
        result = Qwen3LayerConfig(
            vocab_size=_positive_int(config, "vocab_size"),
            num_hidden_layers=_positive_int(config, "num_hidden_layers"),
            hidden_size=_positive_int(config, "hidden_size"),
            intermediate_size=_positive_int(config, "intermediate_size"),
            num_attention_heads=_positive_int(config, "num_attention_heads"),
            num_key_value_heads=_positive_int(config, "num_key_value_heads"),
            head_dim=_positive_int(config, "head_dim"),
            epsilon=float(config.get("rms_norm_eps", 1e-6)),
            rope_theta=float(config.get("rope_theta", 1_000_000.0)),
            max_position_embeddings=_positive_int(config, "max_position_embeddings"),
            padding_idx=_optional_int(config.get("pad_token_id")),
            tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        )
        if result.query_size != result.hidden_size:
            raise ImporterError("Qwen3 P0 requires num_attention_heads * head_dim == hidden_size.")
        if result.num_attention_heads % result.num_key_value_heads:
            raise ImporterError("Qwen3 attention heads must be divisible by KV heads.")
        if result.epsilon <= 0 or result.rope_theta <= 0:
            raise ImporterError("Qwen3 RMS epsilon and RoPE theta must be positive.")
        return result


class Qwen3ModelImporter:
    """Import a complete dense Qwen3 causal-LM decode step.

    The importer establishes the reusable function boundary directly: ``main``
    owns embedding, per-layer weights, LM head and sampling, while every layer
    calls one semantic ``decode_layer`` function.  Middle-end passes must
    preserve and refine that boundary; codegen is not allowed to rediscover it
    from a flattened graph.
    """

    def __init__(
        self,
        checkpoint: Checkpoint | str,
        *,
        block_size: int = 256,
        num_blocks: int = 16,
        revision: str | None = None,
    ) -> None:
        self.checkpoint = (
            DirectoryCheckpoint(checkpoint) if isinstance(checkpoint, str) else checkpoint
        )
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.revision = revision
        self.config = Qwen3LayerImporter._parse_config(self.checkpoint.config)
        self.layer_prefixes = tuple(
            Qwen3LayerImporter._discover_layer_prefix(self.checkpoint.keys, layer)
            for layer in range(self.config.num_hidden_layers)
        )
        model_prefixes = {
            prefix[:-len(f"layers.{layer}.")]
            for layer, prefix in enumerate(self.layer_prefixes)
        }
        if len(model_prefixes) != 1:
            raise ImporterError(
                f"Qwen3 decoder layers do not share one model namespace: {sorted(model_prefixes)}."
            )
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
                "architecture": "Qwen3ForCausalLM",
                "model_type": "qwen3",
                "layer": None,
                "num_layers": config.num_hidden_layers,
                "mode": "decode-1",
                "output_boundary": "logits_fp32_and_greedy_token",
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
                "max_position_embeddings": config.max_position_embeddings,
                "tie_word_embeddings": config.tie_word_embeddings,
                "paged_attention": {
                    "block_size": self.block_size,
                    "num_blocks": self.num_blocks,
                    "cache_layout": [
                        "NumBlocks", "NumLayers", "KV", "BlockSize", "NumKVHeads", "HeadDim",
                    ],
                    "vectorized_axes": ["HeadDim"],
                    "lanes": [8],
                },
            },
        )
        hidden_type = tensor_type(DType.BFLOAT16, [1, config.hidden_size])
        intermediate_type = tensor_type(DType.BFLOAT16, [1, config.intermediate_size])
        logits_bf16_type = tensor_type(DType.BFLOAT16, [1, config.vocab_size])
        logits_type = tensor_type(DType.FLOAT32, [1, config.vocab_size])
        token_type = tensor_type(DType.INT32, [1])
        state_type = self.state_config.ref_type
        input_ids = builder.var("input_ids", tensor_type(DType.INT32, [1]), id="input_ids")
        state = builder.var(
            "paged_attention_kv_cache", state_type, id="paged_attention_kv_cache"
        )
        entry_state = state

        def weight_at(
            name: str,
            key: str,
            shape: tuple[int, ...],
            *,
            node_id: str,
            rdata_group: Mapping[str, object] | None = None,
        ):
            info = self._require_tensor(key, shape, DType.BFLOAT16)
            return builder.weight(
                name,
                tensor_type(info.dtype, info.shape),
                source=info.source,
                key=key,
                id=node_id,
                metadata=(
                    {}
                    if rdata_group is None
                    else {"rdata_group": dict(rdata_group)}
                ),
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

        # nncase/HuggingFace: positions belong to the token step and are
        # shared across layers. Cache writes do not advance the step until
        # the final decoder call, so this read must precede that call chain.
        position_embeddings = _build_rotary_embedding(
            builder, prefix="main", value=hidden, state=state, config=config,
        )
        decode_cos = decode_parameter("rotary_cos", position_embeddings[0].type)
        decode_sin = decode_parameter("rotary_sin", position_embeddings[1].type)

        decode_hidden = decode_parameter("hidden", hidden_type)
        decode_state = decode_parameter("state", state_type)
        decode_input_norm_weight = decode_parameter(
            "input_norm_weight", tensor_type(DType.BFLOAT16, [config.hidden_size])
        )
        decode_q_weight = decode_parameter(
            "q_weight", tensor_type(DType.BFLOAT16, [config.query_size, config.hidden_size])
        )
        decode_k_weight = decode_parameter(
            "k_weight", tensor_type(DType.BFLOAT16, [config.kv_size, config.hidden_size])
        )
        decode_v_weight = decode_parameter(
            "v_weight", tensor_type(DType.BFLOAT16, [config.kv_size, config.hidden_size])
        )
        decode_q_norm_weight = decode_parameter(
            "q_norm_weight", tensor_type(DType.BFLOAT16, [config.head_dim])
        )
        decode_k_norm_weight = decode_parameter(
            "k_norm_weight", tensor_type(DType.BFLOAT16, [config.head_dim])
        )
        decode_o_weight = decode_parameter(
            "o_weight", tensor_type(DType.BFLOAT16, [config.hidden_size, config.query_size])
        )
        decode_post_norm_weight = decode_parameter(
            "post_norm_weight", tensor_type(DType.BFLOAT16, [config.hidden_size])
        )
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
        decode_layer_id = decode_parameter(
            "layer_id", tensor_type(DType.INT32, [])
        )
        decode_advance_sequence = decode_parameter(
            "advance_sequence", tensor_type(DType.BOOL, [])
        )
        decode_parameters = (
            decode_hidden,
            decode_state,
            decode_input_norm_weight,
            decode_q_weight,
            decode_k_weight,
            decode_v_weight,
            decode_q_norm_weight,
            decode_k_norm_weight,
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
            q_norm_weight=decode_q_norm_weight,
            k_norm_weight=decode_k_norm_weight,
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
            attrs={"activation": "silu"},
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

            def layer_weight(name: str, shape: tuple[int, ...]):
                return weight_at(
                    f"layers.{layer}.{name}",
                    layer_prefix + name,
                    shape,
                    node_id=f"w_{prefix}_{name.replace('.', '_')}",
                    rdata_group={
                        "name": f"qwen3.layer.{name}",
                        "index": layer,
                        "count": config.num_hidden_layers,
                    },
                )

            input_norm_weight = layer_weight(
                "input_layernorm.weight", (config.hidden_size,)
            )
            q_weight = layer_weight(
                "self_attn.q_proj.weight", (config.query_size, config.hidden_size)
            )
            k_weight = layer_weight(
                "self_attn.k_proj.weight", (config.kv_size, config.hidden_size)
            )
            v_weight = layer_weight(
                "self_attn.v_proj.weight", (config.kv_size, config.hidden_size)
            )
            q_norm_weight = layer_weight("self_attn.q_norm.weight", (config.head_dim,))
            k_norm_weight = layer_weight("self_attn.k_norm.weight", (config.head_dim,))
            o_weight = layer_weight(
                "self_attn.o_proj.weight", (config.hidden_size, config.query_size)
            )
            post_norm_weight = layer_weight(
                "post_attention_layernorm.weight", (config.hidden_size,)
            )
            gate_weight = layer_weight(
                "mlp.gate_proj.weight", (config.intermediate_size, config.hidden_size)
            )
            up_weight = layer_weight(
                "mlp.up_proj.weight", (config.intermediate_size, config.hidden_size)
            )
            down_weight = layer_weight(
                "mlp.down_proj.weight", (config.hidden_size, config.intermediate_size)
            )
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
                    q_norm_weight,
                    k_norm_weight,
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
                id=(
                    "updated_state"
                    if layer + 1 == config.num_hidden_layers
                    else f"{prefix}_updated_state"
                ),
                attrs={"index": 1},
            )

        final_norm_weight = weight_at(
            "norm.weight",
            self.model_prefix + "norm.weight",
            (config.hidden_size,),
            node_id="w_norm_weight",
        )
        final_hidden = builder.call(
            "nn.rms_norm",
            [hidden, final_norm_weight],
            hidden_type,
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
            logits_bf16_type,
            id="lm_head",
            attrs={"transpose_a": False, "transpose_b": True},
        )
        logits = builder.call(
            "tensors.cast",
            [logits_bf16],
            logits_type,
            id="logits",
            attrs={"dtype": DType.FLOAT32.value},
        )
        next_token = builder.call(
            "nn.greedy_sample", [logits], token_type, id="next_token"
        )
        builder.function(
            "main", [input_ids, entry_state], [logits, next_token, state]
        )
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
        return verify_module(attach_import_source_locations(
            builder.build(entry="main"),
            architecture="Qwen3ForCausalLM",
            revision=self.revision,
        ))

    def _require_tensor(self, key: str, shape: tuple[int, ...], dtype: DType) -> TensorInfo:
        info = self.checkpoint.tensor_info(key)
        if info.shape != shape:
            raise ImporterError(f"Tensor {key!r} must have shape {shape}, got {info.shape}.")
        if info.dtype != dtype:
            raise ImporterError(f"Tensor {key!r} must use {dtype.value}, got {info.dtype.value}.")
        return info


def import_qwen3_layer(
    checkpoint: Checkpoint | str,
    *,
    layer: int = 0,
    block_size: int = 256,
    num_blocks: int = 16,
    revision: str | None = None,
) -> IRModule:
    return Qwen3LayerImporter(
        checkpoint, layer=layer, block_size=block_size, num_blocks=num_blocks, revision=revision).import_module()


def import_qwen3_model(
    checkpoint: Checkpoint | str,
    *,
    block_size: int = 256,
    num_blocks: int = 16,
    revision: str | None = None,
) -> IRModule:
    return Qwen3ModelImporter(
        checkpoint,
        block_size=block_size,
        num_blocks=num_blocks,
        revision=revision,
    ).import_module()


def _positive_int(config: Mapping[str, Any], key: str) -> int:
    try:
        value = int(config[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ImporterError(f"Qwen3 config field {key!r} must be a positive integer.") from error
    if value <= 0:
        raise ImporterError(f"Qwen3 config field {key!r} must be positive.")
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ImporterError("Qwen3 pad_token_id must be an integer or null.")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ImporterError("Qwen3 pad_token_id must be an integer or null.") from error


__all__ = [
    "Qwen3LayerConfig",
    "Qwen3LayerImporter",
    "Qwen3ModelImporter",
    "import_qwen3_layer",
    "import_qwen3_model",
]
