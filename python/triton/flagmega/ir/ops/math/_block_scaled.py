# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Shared PyTorch semantics for block-scaled FP8 math ops."""

from __future__ import annotations

from triton.flagmega.errors import EvaluationError


def dynamic_block_quant_dequant(value, *, block_k: int):
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
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax == 0, torch.ones_like(amax), amax / fp8_max)
    quantized = torch.clamp(grouped / scale, -fp8_max, fp8_max).to(dtype=torch.float8_e4m3fn)
    dequantized = quantized.to(dtype=torch.float32) * scale
    return dequantized.reshape(*original_shape[:-1], padded_reduction)[..., :reduction]


def block_scaled_linear(value, weight, weight_scale_inv, *, block_n: int, block_k: int, output_dtype=None):
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
    activation = dynamic_block_quant_dequant(value, block_k=block_k)
    expanded_scale = weight_scale_inv.to(dtype=torch.float32).repeat_interleave(block_n, dim=0).repeat_interleave(
        block_k, dim=1)[:n, :k]
    dequantized_weight = weight.to(dtype=torch.float32) * expanded_scale
    output = torch.matmul(activation, dequantized_weight.transpose(0, 1))
    return output.to(dtype=output_dtype or value.dtype)


def _torch():
    try:
        import torch
    except ImportError as error:
        raise EvaluationError("Block-scaled op evaluation requires PyTorch.") from error
    return torch


__all__ = ["block_scaled_linear", "dynamic_block_quant_dequant"]
