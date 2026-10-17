# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Shared PyTorch semantics for block-scaled FP8 math ops."""

from __future__ import annotations

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir.types import DType, VectorType


def normalize_block_scaled_precision(attrs, opname):
    """Normalize optional computation precision independently of FP8 storage."""
    quantization = attrs.get("activation_quantization", "dynamic_fp8")
    if not isinstance(quantization, str) or quantization not in {"dynamic_fp8", "none", "dynamic_fp8_sm90"}:
        raise IRSchemaError(f"{opname} activation_quantization must be dynamic_fp8, none or dynamic_fp8_sm90.")
    dtype = attrs.get("weight_dequant_dtype")
    if dtype is not None:
        try:
            dtype = DType(dtype)
        except (ValueError, TypeError) as error:
            raise IRSchemaError(f"{opname} weight_dequant_dtype must be float16 or bfloat16.") from error
        if dtype not in {DType.FLOAT16, DType.BFLOAT16}:
            raise IRSchemaError(f"{opname} weight_dequant_dtype must be float16 or bfloat16.")
        dtype = dtype.value
    result = {}
    if quantization != "dynamic_fp8":
        result["activation_quantization"] = quantization
    if dtype is not None:
        result["weight_dequant_dtype"] = dtype
    if result and attrs.get("weight_encoding") is not None:
        raise IRSchemaError(f"{opname} precision overrides require FP8 storage, not a native GGUF encoding.")
    return result


def verify_block_scaled_precision_inputs(value, weight, scale, attrs, opname):
    if attrs.get("activation_quantization", "dynamic_fp8") == "dynamic_fp8" and attrs.get("weight_dequant_dtype") is None:
        return
    scalar_weight = weight.dtype.elem_type if isinstance(weight.dtype, VectorType) else weight.dtype
    if (value.dtype not in {DType.FLOAT16, DType.BFLOAT16, DType.FLOAT32}
            or scalar_weight != DType.FLOAT8_E4M3FN or scale.dtype != DType.FLOAT32):
        raise IRSchemaError(f"{opname} precision overrides require scalar F16/BF16/F32 activations, "
                            "FP8 E4M3FN weights and FP32 scales.")
    if attrs.get("activation_quantization") == "dynamic_fp8_sm90" and value.dtype != DType.BFLOAT16:
        raise IRSchemaError(f"{opname} SM90 reciprocal activation quantization requires BF16 input.")


def normalize_glu_activation_quantization(attrs, opname):
    """GLU shares activation quantization but retains its own weight contract."""
    result = normalize_block_scaled_precision(
        {"activation_quantization": attrs.get("activation_quantization", "dynamic_fp8"),
         "weight_encoding": attrs.get("gate_weight_encoding") or attrs.get("up_weight_encoding")}, opname)
    if result.get("activation_quantization") == "none":
        raise IRSchemaError(f"{opname} activation_quantization must be dynamic_fp8 or dynamic_fp8_sm90.")
    return result


def block_scaled_precision_supports_target(attrs, platform, architecture):
    return (attrs.get("activation_quantization") != "dynamic_fp8_sm90"
            or (platform == "nvidia" and architecture == "sm90"))


def block_scaled_mma_supports_precision(attrs, value_dtype, *, block_k):
    if (attrs.get("activation_quantization", "dynamic_fp8") in {"dynamic_fp8", "dynamic_fp8_sm90"}
            and attrs.get("weight_dequant_dtype") is None):
        return True
    dtype = attrs.get("weight_dequant_dtype")
    return (attrs.get("activation_quantization") == "none" and dtype in {"float16", "bfloat16"}
            and DType(value_dtype).value == dtype and block_k >= 16)


def normalize_output_lanes(lanes, opname):
    """Validate optional final-N output packing without choosing a width."""
    if not isinstance(lanes, (tuple, list)) or any(
            isinstance(lane, bool) or not isinstance(lane, int) or lane <= 0 for lane in lanes):
        raise IRSchemaError(f"{opname} output_lanes must be a sequence of positive integers.")
    return tuple(lanes)


def dynamic_block_quant_dequant(value, *, block_k: int, mode="dynamic_fp8"):
    torch = _torch()
    if block_k <= 0:
        raise EvaluationError(f"Dynamic FP8 block K must be positive, got {block_k}.")
    original_shape = tuple(value.shape)
    reduction = original_shape[-1]
    groups = (reduction + block_k - 1) // block_k
    padded_reduction = groups * block_k
    fp32 = value.to(dtype=torch.float32)
    if padded_reduction != reduction:
        fp32 = torch.nn.functional.pad(fp32, (0, padded_reduction - reduction))
    grouped = fp32.reshape(*original_shape[:-1], groups, block_k)
    if mode == "dynamic_fp8_ue8m0":
        from triton.flagmega.ir.ops.math._ue8m0_quantization import ue8m0_quantize
        quantized, scale = ue8m0_quantize(grouped)
        return (quantized.float() * scale).reshape(*original_shape[:-1], padded_reduction)[..., :reduction]
    if mode == "dynamic_fp8_sm90":
        from triton.flagmega.ir.ops.math._sm90_quantization import sm90_bf16_quantize
        quantized, scale = sm90_bf16_quantize(grouped.bfloat16() if value.dtype == torch.bfloat16 else grouped)
        return (quantized.float() * scale).reshape(*original_shape[:-1], padded_reduction)[..., :reduction]
    if mode != "dynamic_fp8":
        raise EvaluationError(f"Unsupported activation quantization mode {mode!r}.")
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    # CUDA tensor/scalar division may multiply by a rounded reciprocal. Keep
    # the divisor on the value's device so exact FP8 ties use the same division
    # as the CPU semantics and the generated kernels' div_rn.
    divisor = amax.new_tensor(fp8_max)
    scale = torch.where(amax == 0, torch.ones_like(amax), amax / divisor)
    quantized = torch.clamp(grouped / scale, -fp8_max, fp8_max).to(dtype=torch.float8_e4m3fn)
    dequantized = quantized.to(dtype=torch.float32) * scale
    return dequantized.reshape(*original_shape[:-1], padded_reduction)[..., :reduction]


def block_scaled_linear(value, weight, weight_scale_inv, *, block_n: int, block_k: int, output_dtype=None,
                        activation_quantization="dynamic_fp8", weight_dequant_dtype=None):
    torch = _torch()
    if value.ndim < 2 or weight.ndim != 2 or weight_scale_inv.ndim != 2:
        raise EvaluationError("Block-scaled linear requires value rank >=2 and rank-2 weight/scale.")
    n, k = weight.shape
    if value.shape[-1] != k:
        raise EvaluationError(f"Block-scaled linear K mismatch: value {value.shape[-1]}, weight {k}.")
    expected_scale = ((n + block_n - 1) // block_n, (k + block_k - 1) // block_k)
    if tuple(weight_scale_inv.shape) != expected_scale:
        raise EvaluationError(
            f"Block-scaled weight scale must have shape {expected_scale}, got {tuple(weight_scale_inv.shape)}.")
    activation = (value.float() if activation_quantization == "none" else
                  dynamic_block_quant_dequant(value, block_k=block_k, mode=activation_quantization))
    expanded_scale = weight_scale_inv.to(dtype=torch.float32).repeat_interleave(block_n, dim=0).repeat_interleave(
        block_k, dim=1)[:n, :k]
    dequantized_weight = weight.to(dtype=torch.float32) * expanded_scale
    if weight_dequant_dtype is not None:
        dequantized_weight = dequantized_weight.to(getattr(torch, weight_dequant_dtype)).float()
    output = torch.matmul(activation, dequantized_weight.transpose(0, 1))
    return output.to(dtype=output_dtype or value.dtype)


def _torch():
    try:
        import torch
    except ImportError as error:
        raise EvaluationError("Block-scaled op evaluation requires PyTorch.") from error
    return torch


__all__ = ["block_scaled_linear", "dynamic_block_quant_dequant"]
