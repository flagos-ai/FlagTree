# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fold inverse views and collapse contiguous final-axis repacking to a view."""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import IRModule, Node, get_definition
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.tensors.pack import _pack_axes
from triton.flagmega.ir.types import data_type_to_data
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
    view = result["bitcast"]
    view_type = tensor_of(view.type)
    # Packing a non-final axis permutes coordinates. Final-axis packing
    # prepends lanes directly before existing lanes, preserving flat storage
    # order even when a Bitcast restored only some of the original lanes.
    # Packs carry either the multi-axis ``axes`` form or the single ``axis``
    # form with one lane per repetition; _pack_axes normalizes both.
    axes = _pack_axes(node.attrs, len(node.attrs["lanes"]), view_type.rank)
    if not axes or any(axis != view_type.rank - 1 for axis in axes):
        return node
    if node.type == source.type:
        return RewriteRedirect(source.id)
    try:
        prepared = get_definition("tensors.bitcast").prepare(
            (source,), {"dtype": data_type_to_data(tensor_of(node.type).dtype)})
    except IRSchemaError:
        return node
    if prepared.result_type != node.type:
        return node
    return Node(node.id, "tensors.bitcast", (source.id,), prepared.result_type,
                prepared.effect, prepared.attrs, {**dict(node.metadata), "rewritten_by": "FoldPackBitcast"})


def fold_pack_bitcast_rule() -> RewriteRule:
    return RewriteRule("FoldPackBitcast", pattern=_PATTERN, rewrite=_rewrite)


__all__ = ["fold_pack_bitcast_rule"]
