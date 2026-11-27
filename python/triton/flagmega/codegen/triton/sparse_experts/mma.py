# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Shared candidate and resumed-TIR checks for block-scaled FP8 WGMMA."""

from triton.flagmega.ir import SBP, DistributedType, DType, VectorType, logical_type


def supports_block_fp8_wgmma(op, types, attrs):
    if op not in {"nn.sparse_experts_gate_up", "nn.partitioned_sparse_experts_down"}:
        return False
    if (attrs.get("activation_quantization", "dynamic_fp8") != "dynamic_fp8_ue8m0"
            or attrs.get("weight_block_n", 128) != 128 or attrs.get("weight_block_k", 128) != 128):
        return False
    activation_name = "dispatched" if op == "nn.sparse_experts_gate_up" else "activations"
    activation = logical_type(types[activation_name])
    dtype = activation.dtype.elem_type if isinstance(activation.dtype, VectorType) else activation.dtype
    if dtype != DType.BFLOAT16:
        return False
    fields = ("gate", "up") if op == "nn.sparse_experts_gate_up" else ("down",)
    for field in fields:
        value = types[field + "_weight"]
        weight = logical_type(value)
        scale = logical_type(types[field + "_proj_scale"])
        if (weight.dtype != DType.FLOAT8_E4M3FN or scale.dtype != DType.FLOAT32
                or scale.rank != weight.rank or not weight.shape[-1].is_fixed
                or weight.shape[-1].fixed_value % 128):
            return False
        if isinstance(value, DistributedType) and value.axis_policies[-1] != SBP.broadcast():
            return False
    return True
