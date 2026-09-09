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


def decompose_complex_ops(module: IRModule) -> IRModule:
    """Expose independently optimizable semantic stages without target policy."""

    current = DataflowPass(
        "DecomposeComplexOps",
        (decompose_layer_norm_rule(), decompose_rms_norm_rule(),
         fuse_wide_glu_rule(), decompose_sparse_experts_rule(),
         fuse_norm_apply_cast_rule(), form_qkv_rope_with_cache_rule()),
    ).run(verify_module(module))
    return decompose_gated_delta_net(current)


__all__ = ["decompose_complex_ops"]
