# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Remove the distribution-only NormStats binding after layout selection."""

from triton.flagmega.ir import IRModule, Node
from triton.flagmega.pattern_match import F, wildcard
from triton.flagmega.rules import RewriteRedirect, RewriteRule


def fold_bind_norm_stats_rule() -> RewriteRule:
    pattern = F.nn.is_bind_norm_stats(
        wildcard("input"),
        wildcard("stats"),
        target_name="target",
        call_name="call",
    )

    def rewrite(result, module: IRModule):
        del module
        call = result["call"]
        stats = result["stats"]
        assert isinstance(call, Node) and isinstance(stats, Node)
        return RewriteRedirect(stats.id)

    return RewriteRule("FoldBindNormStats", pattern, rewrite)


__all__ = ["fold_bind_norm_stats_rule"]
