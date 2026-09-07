# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Expose RMSNorm as generic NormStats/NormApply dataflow."""

from __future__ import annotations

from triton.flagmega.ir import IRModule, Node, TensorType
from triton.flagmega.ir.ops.nn.rms_norm import RMSNorm
from triton.flagmega.pattern_match import F, wildcard
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.neutral._utility import decomposition_metadata, make_node


def decompose_rms_norm_rule() -> RewriteRule:
    pattern = F.nn.is_rms_norm(
        wildcard("input", type_pattern=RMSNorm.value.type_pattern),
        wildcard("weight", type_pattern=RMSNorm.weight.type_pattern),
        target_name="target",
        call_name="call",
    )

    def rewrite(result, module: IRModule) -> RewriteResult:
        del module
        source = result["call"]
        value = result["input"]
        weight = result["weight"]
        assert isinstance(source, Node) and isinstance(value, Node) and isinstance(weight, Node)
        if not isinstance(weight.type, TensorType):
            raise TypeError("DecomposeRMSNorm expects a logical tensor weight before AutoDistributed.")
        metadata = decomposition_metadata(source, "DecomposeRMSNorm")
        prefix: list[Node] = []
        scale = weight
        weight_bias = float(source.attrs["weight_bias"])
        if weight_bias != 0.0:
            scale_offset = make_node(
                "builtin.splat_const",
                f"{source.id}.decomposed.scale_offset",
                (),
                {"result_type": weight.type, "value": weight_bias},
                metadata,
            )
            scale = make_node(
                "math.add",
                f"{source.id}.decomposed.scale",
                (weight, scale_offset),
                {},
                metadata,
            )
            prefix.extend((scale_offset, scale))
        bias = make_node(
            "builtin.splat_const",
            f"{source.id}.decomposed.bias",
            (),
            {"result_type": weight.type, "value": 0.0},
            metadata,
        )
        stats = make_node(
            "nn.norm_stats",
            f"{source.id}.decomposed.stats",
            (value,),
            {"axis": -1, "use_mean": False},
            metadata,
        )
        prefix.extend((bias, stats))
        replacement = make_node(
            "nn.norm_apply",
            source.id,
            (value, stats, scale, bias),
            {"axis": -1, "epsilon": float(source.attrs["epsilon"]), "use_mean": False,
             "round_before_scale": True},
            metadata,
        )
        return RewriteResult(replacement, tuple(prefix))

    return RewriteRule("DecomposeRMSNorm", pattern, rewrite)


__all__ = ["decompose_rms_norm_rule"]
