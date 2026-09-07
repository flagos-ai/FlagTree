# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Decompose fused LayerNorm to explicit NormStats and NormApply."""

from __future__ import annotations

from triton.flagmega.ir import IRModule, Node
from triton.flagmega.ir.ops.nn.layer_norm import LayerNorm
from triton.flagmega.pattern_match import F, wildcard
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.neutral._utility import decomposition_metadata, make_node


def decompose_layer_norm_rule() -> RewriteRule:
    pattern = F.nn.is_layer_norm(
        wildcard("input", type_pattern=LayerNorm.value.type_pattern),
        wildcard("scale", type_pattern=LayerNorm.scale.type_pattern),
        wildcard("bias", type_pattern=LayerNorm.bias.type_pattern),
        target_name="target",
        call_name="call",
    )

    def rewrite(result, module: IRModule) -> RewriteResult:
        del module
        source = result["call"]
        assert isinstance(source, Node)
        value = result["input"]
        scale = result["scale"]
        bias = result["bias"]
        assert all(isinstance(item, Node) for item in (value, scale, bias))
        metadata = decomposition_metadata(source, "DecomposeLayerNorm")
        stats = make_node(
            "nn.norm_stats",
            f"{source.id}.decomposed.stats",
            (value,),
            {"axis": int(source.attrs["axis"]), "use_mean": bool(source.attrs["use_mean"])},
            metadata,
        )
        replacement = make_node(
            "nn.norm_apply",
            source.id,
            (value, stats, scale, bias),
            dict(source.attrs),
            metadata,
        )
        return RewriteResult(replacement, (stats,))

    return RewriteRule("DecomposeLayerNorm", pattern, rewrite)


__all__ = ["decompose_layer_norm_rule"]
