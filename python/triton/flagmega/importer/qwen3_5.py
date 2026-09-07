# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Qwen3.5 architecture importer, including Qwen3.8 layer 0."""

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
    TupleType,
    effect,
    tensor_type,
    verify_module,
)
from triton.flagmega.ir.ops.nn._gdn_state import gdn_state_config


@dataclass(frozen=True)
class Qwen35LayerConfig:
    vocab_size: int
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel_size: int
    epsilon: float
    weight_block_n: int
    weight_block_k: int
    padding_idx: int | None

    @property
    def key_dim(self) -> int:
        return self.num_key_heads * self.key_head_dim

    @property
    def value_dim(self) -> int:
        return self.num_value_heads * self.value_head_dim

    @property
    def conv_dim(self) -> int:
        return (2 * self.key_dim) + self.value_dim


class Qwen35Layer0Importer:
    """Import only the first Qwen3.5/Qwen3.8 linear-attention decoder block."""

    def __init__(self, checkpoint: Checkpoint | str, *, revision: str | None = None) -> None:
        self.checkpoint = DirectoryCheckpoint(checkpoint) if isinstance(checkpoint, str) else checkpoint
        self.revision = revision
        self.config = self._parse_config(self.checkpoint.config)
        self.layer_prefix = self._discover_layer_prefix(self.checkpoint.keys)
        self.model_prefix = self._model_prefix(self.layer_prefix)

    def import_module(self) -> IRModule:
        config = self.config
        builder = IRBuilder(
            dialect="high_level",
            stage="imported",
            metadata={
                "architecture": "Qwen3_5ForConditionalGeneration",
                "model_type": "qwen3_5",
                "layer": 0,
                "mode": "decode-1",
                "revision": self.revision,
                "layer_prefix": self.layer_prefix,
                "model_prefix": self.model_prefix,
                "vocab_size": config.vocab_size,
                "padding_idx": config.padding_idx,
                "num_hidden_layers": config.num_hidden_layers,
                "hidden_size": config.hidden_size,
                "intermediate_size": config.intermediate_size,
                "weight_block_size": [config.weight_block_n, config.weight_block_k],
            },
        )
        input_ids_type = tensor_type(DType.INT32, [1])
        hidden_type = tensor_type(DType.BFLOAT16, [1, config.hidden_size])
        state_type = gdn_state_config(config).ref_type
        input_ids = builder.var("input_ids", input_ids_type, id="input_ids")
        state = builder.var("gated_delta_net_state", state_type, id="gdn_state")

        def weight_at(
            name: str,
            key: str,
            expected_shape: tuple[int, ...],
            *,
            expected_dtype: DType | tuple[DType, ...] | None = None,
        ):
            info = self._require_tensor(key, expected_shape, expected_dtype)
            value_type = tensor_type(info.dtype, info.shape)
            return builder.weight(name, value_type, source=info.source, key=key, id=_weight_id(name))

        def weight(
            relative_name: str,
            expected_shape: tuple[int, ...],
            *,
            expected_dtype: DType | tuple[DType, ...] | None = None,
        ):
            return weight_at(
                relative_name,
                f"{self.layer_prefix}{relative_name}",
                expected_shape,
                expected_dtype=expected_dtype,
            )

        embedding_weight = weight_at(
            "embed_tokens.weight",
            f"{self.model_prefix}embed_tokens.weight",
            (config.vocab_size, config.hidden_size),
            expected_dtype=DType.BFLOAT16,
        )
        hidden = builder.call(
            "nn.embedding",
            [input_ids, embedding_weight],
            hidden_type,
            id="token_embedding",
            attrs={"padding_idx": config.padding_idx},
        )

        input_norm_weight = weight("input_layernorm.weight", (config.hidden_size, ), expected_dtype=DType.BFLOAT16)
        normalized = builder.call(
            "nn.rms_norm",
            [hidden, input_norm_weight],
            hidden_type,
            id="input_norm",
            attrs={"epsilon": config.epsilon, "weight_bias": 1.0},
        )

        qkv_weight = weight(
            "linear_attn.in_proj_qkv.weight",
            (config.conv_dim, config.hidden_size),
            expected_dtype=DType.FLOAT8_E4M3FN,
        )
        qkv_scale = weight(
            "linear_attn.in_proj_qkv.weight_scale_inv",
            (_ceil_div(config.conv_dim, config.weight_block_n), _ceil_div(config.hidden_size, config.weight_block_k)),
            expected_dtype=(DType.BFLOAT16, DType.FLOAT32),
        )
        z_weight = weight(
            "linear_attn.in_proj_z.weight",
            (config.value_dim, config.hidden_size),
            expected_dtype=DType.FLOAT8_E4M3FN,
        )
        z_scale = weight(
            "linear_attn.in_proj_z.weight_scale_inv",
            (_ceil_div(config.value_dim, config.weight_block_n), _ceil_div(config.hidden_size, config.weight_block_k)),
            expected_dtype=(DType.BFLOAT16, DType.FLOAT32),
        )
        b_weight = weight(
            "linear_attn.in_proj_b.weight",
            (config.num_value_heads, config.hidden_size),
            expected_dtype=DType.BFLOAT16,
        )
        a_weight = weight(
            "linear_attn.in_proj_a.weight",
            (config.num_value_heads, config.hidden_size),
            expected_dtype=DType.BFLOAT16,
        )
        conv_weight = weight(
            "linear_attn.conv1d.weight",
            (config.conv_dim, 1, config.conv_kernel_size),
            expected_dtype=DType.BFLOAT16,
        )
        a_log = weight("linear_attn.A_log", (config.num_value_heads, ))
        dt_bias = weight("linear_attn.dt_bias", (config.num_value_heads, ))
        gdn_norm_weight = weight("linear_attn.norm.weight", (config.value_head_dim, ))
        out_weight = weight(
            "linear_attn.out_proj.weight",
            (config.hidden_size, config.value_dim),
            expected_dtype=DType.FLOAT8_E4M3FN,
        )
        out_scale = weight(
            "linear_attn.out_proj.weight_scale_inv",
            (_ceil_div(config.hidden_size, config.weight_block_n), _ceil_div(config.value_dim, config.weight_block_k)),
            expected_dtype=(DType.BFLOAT16, DType.FLOAT32),
        )
        gdn_type = TupleType((hidden_type, state_type))
        gdn = builder.call(
            "nn.gated_delta_net",
            [
                normalized,
                state,
                qkv_weight,
                qkv_scale,
                z_weight,
                z_scale,
                b_weight,
                a_weight,
                conv_weight,
                a_log,
                dt_bias,
                gdn_norm_weight,
                out_weight,
                out_scale,
            ],
            gdn_type,
            id="gated_delta_net",
            effect=effect("read_write", "gated_delta_net_state"),
            attrs={
                "num_key_heads": config.num_key_heads,
                "num_value_heads": config.num_value_heads,
                "key_head_dim": config.key_head_dim,
                "value_head_dim": config.value_head_dim,
                "conv_kernel_size": config.conv_kernel_size,
                "epsilon": config.epsilon,
                "weight_block_n": config.weight_block_n,
                "weight_block_k": config.weight_block_k,
            },
        )
        gdn_output = builder.call("builtin.get_item", [gdn], hidden_type, id="gdn_output", attrs={"index": 0})
        updated_state = builder.call("builtin.get_item", [gdn], state_type, id="updated_state", attrs={"index": 1})
        after_gdn = builder.call("math.add", [hidden, gdn_output], hidden_type, id="after_gdn")

        post_norm_weight = weight("post_attention_layernorm.weight", (config.hidden_size, ), expected_dtype=DType.BFLOAT16)
        mlp_input = builder.call(
            "nn.rms_norm",
            [after_gdn, post_norm_weight],
            hidden_type,
            id="post_attention_norm",
            attrs={"epsilon": config.epsilon, "weight_bias": 1.0},
        )
        intermediate_type = tensor_type(DType.BFLOAT16, [1, config.intermediate_size])
        gate_weight = weight(
            "mlp.gate_proj.weight", (config.intermediate_size, config.hidden_size), expected_dtype=DType.FLOAT8_E4M3FN)
        gate_scale = weight(
            "mlp.gate_proj.weight_scale_inv",
            (_ceil_div(config.intermediate_size, config.weight_block_n), _ceil_div(config.hidden_size, config.weight_block_k)),
            expected_dtype=(DType.BFLOAT16, DType.FLOAT32),
        )
        up_weight = weight(
            "mlp.up_proj.weight", (config.intermediate_size, config.hidden_size), expected_dtype=DType.FLOAT8_E4M3FN)
        up_scale = weight(
            "mlp.up_proj.weight_scale_inv",
            (_ceil_div(config.intermediate_size, config.weight_block_n), _ceil_div(config.hidden_size, config.weight_block_k)),
            expected_dtype=(DType.BFLOAT16, DType.FLOAT32),
        )
        gated_mlp = builder.call(
            "nn.matmul_glu",
            [mlp_input, gate_weight, up_weight, gate_scale, up_scale],
            intermediate_type,
            id="mlp_gate_up",
            attrs={
                "activation": "silu",
                "weight_block_n": config.weight_block_n,
                "weight_block_k": config.weight_block_k,
            },
        )
        down_weight = weight(
            "mlp.down_proj.weight", (config.hidden_size, config.intermediate_size), expected_dtype=DType.FLOAT8_E4M3FN)
        down_scale = weight(
            "mlp.down_proj.weight_scale_inv",
            (_ceil_div(config.hidden_size, config.weight_block_n), _ceil_div(config.intermediate_size, config.weight_block_k)),
            expected_dtype=(DType.BFLOAT16, DType.FLOAT32),
        )
        mlp_output = builder.call(
            "math.block_scaled_matmul",
            [gated_mlp, down_weight, down_scale],
            hidden_type,
            id="mlp_down",
            attrs={"weight_block_n": config.weight_block_n, "weight_block_k": config.weight_block_k},
        )
        output = builder.call("math.add", [after_gdn, mlp_output], hidden_type, id="output")
        builder.function("main", [input_ids, state], [output, updated_state])
        return verify_module(attach_import_source_locations(
            builder.build(entry="main"),
            architecture="Qwen3_5ForConditionalGeneration",
            revision=self.revision,
        ))

    def _require_tensor(
        self,
        key: str,
        expected_shape: tuple[int, ...],
        expected_dtype: DType | tuple[DType, ...] | None = None,
    ) -> TensorInfo:
        info = self.checkpoint.tensor_info(key)
        if info.shape != expected_shape:
            raise ImporterError(f"Tensor {key!r} must have shape {expected_shape}, got {info.shape}.")
        expected_dtypes = (
            ()
            if expected_dtype is None
            else expected_dtype
            if isinstance(expected_dtype, tuple)
            else (expected_dtype, )
        )
        if expected_dtypes and info.dtype not in expected_dtypes:
            names = ", ".join(dtype.value for dtype in expected_dtypes)
            raise ImporterError(f"Tensor {key!r} must use one of [{names}], got {info.dtype.value}.")
        return info

    @staticmethod
    def _discover_layer_prefix(keys: tuple[str, ...]) -> str:
        suffix = "layers.0.input_layernorm.weight"
        matches = [key for key in keys if key.endswith(suffix)]
        key_set = set(keys)
        qualified = []
        required = (
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.conv1d.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
        )
        for match in matches:
            prefix = match[:-len("input_layernorm.weight")]
            if all(prefix + relative_name in key_set for relative_name in required):
                qualified.append(prefix)
        if len(qualified) != 1:
            raise ImporterError(
                f"Expected exactly one Qwen linear-attention layer-0 namespace ending in {suffix!r}; "
                f"input-norm matches={matches}, qualified prefixes={qualified}.")
        return qualified[0]

    @staticmethod
    def _model_prefix(layer_prefix: str) -> str:
        suffix = "layers.0."
        if not layer_prefix.endswith(suffix):
            raise ImporterError(f"Qwen layer-0 prefix must end in {suffix!r}, got {layer_prefix!r}.")
        return layer_prefix[:-len(suffix)]

    @staticmethod
    def _parse_config(config: Mapping[str, Any]) -> Qwen35LayerConfig:
        model_type = str(config.get("model_type", ""))
        architectures = tuple(str(value) for value in config.get("architectures", ()))
        if model_type != "qwen3_5" and "Qwen3_5ForConditionalGeneration" not in architectures:
            raise ImporterError(
                f"Expected Qwen3.5/Qwen3.8 config, got model_type={model_type!r}, architectures={architectures}.")
        text_config_value = config.get("text_config")
        if text_config_value is None:
            text_config = config
        elif isinstance(text_config_value, Mapping):
            text_config = text_config_value
        else:
            raise ImporterError("Qwen text_config must be an object when present.")
        layer_types = text_config.get("layer_types")
        if not isinstance(layer_types, list) or not layer_types or layer_types[0] not in {"linear", "linear_attention"}:
            raise ImporterError("Qwen layer 0 must be a linear-attention layer.")
        quantization = text_config.get("quantization_config", config.get("quantization_config"))
        if not isinstance(quantization, Mapping):
            raise ImporterError("Qwen3.8 FP8 import requires quantization_config.")
        if str(quantization.get("quant_method", "")).lower() != "fp8":
            raise ImporterError("Qwen3.8 layer import currently requires quant_method='fp8'.")
        if str(quantization.get("activation_scheme", "")).lower() != "dynamic":
            raise ImporterError("Qwen3.8 layer import currently requires dynamic FP8 activations.")
        fmt = str(quantization.get("fmt", "e4m3")).lower()
        if fmt != "e4m3":
            raise ImporterError(f"Qwen3.8 layer import requires E4M3 FP8, got {fmt!r}.")
        block = quantization.get("weight_block_size")
        if not isinstance(block, list) or len(block) != 2:
            raise ImporterError("quantization_config.weight_block_size must be [N, K].")
        result = Qwen35LayerConfig(
            vocab_size=_positive_int(text_config, "vocab_size"),
            num_hidden_layers=_positive_int(text_config, "num_hidden_layers"),
            hidden_size=_positive_int(text_config, "hidden_size"),
            intermediate_size=_positive_int(text_config, "intermediate_size"),
            num_key_heads=_positive_int(text_config, "linear_num_key_heads"),
            num_value_heads=_positive_int(text_config, "linear_num_value_heads"),
            key_head_dim=_positive_int(text_config, "linear_key_head_dim"),
            value_head_dim=_positive_int(text_config, "linear_value_head_dim"),
            conv_kernel_size=_positive_int(text_config, "linear_conv_kernel_dim"),
            epsilon=float(text_config.get("rms_norm_eps", 1e-6)),
            weight_block_n=int(block[0]),
            weight_block_k=int(block[1]),
            padding_idx=_optional_int(text_config.get("pad_token_id", config.get("pad_token_id")), "pad_token_id"),
        )
        if result.num_value_heads % result.num_key_heads != 0:
            raise ImporterError("linear_num_value_heads must be divisible by linear_num_key_heads.")
        if result.conv_kernel_size < 2:
            raise ImporterError("linear_conv_kernel_dim must be at least two.")
        if result.weight_block_n <= 0 or result.weight_block_k <= 0:
            raise ImporterError("FP8 weight block dimensions must be positive.")
        if result.padding_idx is not None and not -result.vocab_size <= result.padding_idx < result.vocab_size:
            raise ImporterError(
                f"Qwen pad_token_id {result.padding_idx} is outside vocabulary size {result.vocab_size}.")
        return result


def import_qwen3_8_layer0(checkpoint: Checkpoint | str, *, revision: str | None = None) -> IRModule:
    return Qwen35Layer0Importer(checkpoint, revision=revision).import_module()


def _positive_int(config: Mapping[str, Any], key: str) -> int:
    try:
        value = int(config[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ImporterError(f"Qwen config field {key!r} must be a positive integer.") from error
    if value <= 0:
        raise ImporterError(f"Qwen config field {key!r} must be positive, got {value}.")
    return value


def _optional_int(value: object, key: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ImporterError(f"Qwen config field {key!r} must be an integer or null.")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ImporterError(f"Qwen config field {key!r} must be an integer or null.") from error


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _weight_id(relative_name: str) -> str:
    return "w_" + relative_name.replace(".", "_")
