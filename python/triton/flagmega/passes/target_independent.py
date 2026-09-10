# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Implemented target-independent semantic decompositions."""

from __future__ import annotations

from triton.flagmega.ir import IRModule, verify_module
from triton.flagmega.passes.gated_delta_net import decompose_gated_delta_net
from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.rules.neutral import (
    decompose_layer_norm_rule,
    decompose_rms_norm_rule,
    decompose_sparse_experts_rule,
    fuse_wide_glu_rule,
    fuse_norm_apply_cast_rule,
)
from triton.flagmega.rules.neutral.form_qkv_rope_with_cache import form_qkv_rope_with_cache_rule
from triton.flagmega.rules.neutral.fold_cast import fold_cast_rule
from triton.flagmega.rules.neutral.fold_matmul_cast import fold_matmul_cast_rule


def decompose_complex_ops(module: IRModule) -> IRModule:
    """Expose independently optimizable semantic stages without target policy."""

    current = DataflowPass(
        "DecomposeComplexOps",
        (fold_cast_rule(), decompose_layer_norm_rule(), decompose_rms_norm_rule(),
         fuse_wide_glu_rule(), decompose_sparse_experts_rule(),
         fuse_norm_apply_cast_rule(), form_qkv_rope_with_cache_rule()),
    ).run(verify_module(module))
    # Form larger semantic regions before canonicalizing their remaining
    # output conversions; local precision changes must not hide those regions.
    current = decompose_gated_delta_net(current)
    return DataflowPass("FoldOutputConversions", (fold_cast_rule(), fold_matmul_cast_rule())).run(current)


__all__ = ["decompose_complex_ops"]
