# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fold a projection from a literal tuple to the selected field."""

from triton.flagmega.ir import IRModule, Node
from triton.flagmega.pattern_match import F, MatchResult
from triton.flagmega.rules import RewriteRedirect, RewriteRule


_PATTERN = F.tensors.is_get_item(
    F.builtin.is_tuple(call_name="tuple"),
    call_name="get_item",
)


def _rewrite(result: MatchResult, _module: IRModule) -> RewriteRedirect:
    get_item = result["get_item"]
    tuple_value = result["tuple"]
    assert isinstance(get_item, Node) and isinstance(tuple_value, Node)
    index = int(get_item.attrs["index"])
    # GetItem's schema and verifier already prove the bound against TupleType;
    # keep a local assertion because the concrete literal is the redirect
    # source used by both data-flow and e-graph providers.
    assert 0 <= index < len(tuple_value.inputs)
    return RewriteRedirect(tuple_value.inputs[index])


def fold_get_item_tuple_rule() -> RewriteRule:
    return RewriteRule("FoldGetItemTuple", pattern=_PATTERN, rewrite=_rewrite)


__all__ = ["fold_get_item_tuple_rule"]
