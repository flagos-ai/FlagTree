# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Collapse nested storage Bitcasts into one direct reinterpretation."""

from triton.flagmega.ir import IRModule, Node, get_definition
from triton.flagmega.pattern_match import F, MatchResult, wildcard
from triton.flagmega.rules import RewriteRule


_INPUT = wildcard("input")
_PATTERN = F.tensors.is_bitcast(
    F.tensors.is_bitcast(_INPUT, call_name="inner"),
    call_name="outer",
)


def _rewrite(result: MatchResult, _module: IRModule) -> Node:
    node = result["outer"]
    source = result["input"]
    assert isinstance(node, Node) and isinstance(source, Node)
    prepared = get_definition("tensors.bitcast").prepare((source,), node.attrs)
    if prepared.result_type != node.type:
        return node
    return Node(
        node.id,
        "tensors.bitcast",
        (source.id,),
        prepared.result_type,
        prepared.effect,
        prepared.attrs,
        {**dict(node.metadata), "rewritten_by": "FoldBitcastBitcast"},
    )


def fold_bitcast_bitcast_rule() -> RewriteRule:
    return RewriteRule("FoldBitcastBitcast", pattern=_PATTERN, rewrite=_rewrite)


__all__ = ["fold_bitcast_bitcast_rule"]
