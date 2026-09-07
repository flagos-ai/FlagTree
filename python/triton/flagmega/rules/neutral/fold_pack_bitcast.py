# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fold Pack(Bitcast(x)) when the composition restores x's type."""

from triton.flagmega.ir import IRModule, Node
from triton.flagmega.pattern_match import F, MatchResult, wildcard
from triton.flagmega.rules import RewriteRedirect, RewriteRule


_INPUT = wildcard("input")
_PATTERN = F.tensors.is_pack(
    F.tensors.is_bitcast(_INPUT, call_name="bitcast"),
    call_name="pack",
)


def _rewrite(result: MatchResult, _module: IRModule):
    node = result["pack"]
    source = result["input"]
    assert isinstance(node, Node) and isinstance(source, Node)
    return RewriteRedirect(source.id) if node.type == source.type else node


def fold_pack_bitcast_rule() -> RewriteRule:
    return RewriteRule("FoldPackBitcast", pattern=_PATTERN, rewrite=_rewrite)


__all__ = ["fold_pack_bitcast_rule"]
