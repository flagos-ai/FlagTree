# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Expose split-KV paged-attention states before distributed search."""

from __future__ import annotations

from triton.flagmega.ir import IRModule, Node
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.ntt.paged_attention_partial import (
    PagedAttentionPartial,
)
from triton.flagmega.pattern_match import F, wildcard
from triton.flagmega.rules import (
    RewriteEffectPolicy,
    RewriteResult,
    RewriteRule,
)
from triton.flagmega.rules.neutral._utility import (
    decomposition_metadata,
    make_node,
)


def paged_attention_split_plan(placements):
    """Choose nncase's smallest common physical-block hierarchy axis."""

    placements = tuple(placements)
    if not placements:
        raise ValueError(
            "PyNTT paged-attention decomposition requires at least one placement."
        )
    rank = placements[0].rank
    if any(placement.rank != rank for placement in placements):
        raise ValueError(
            "PyNTT paged-attention decomposition requires equal placement ranks."
        )
    candidates: list[tuple[int, int]] = []
    for axis in range(rank):
        if not all(
            placement.is_physical_block_axis(axis) for placement in placements
        ):
            continue
        extents = {placement.hierarchy[axis] for placement in placements}
        if len(extents) != 1:
            continue
        extent = next(iter(extents))
        if extent > 1:
            candidates.append((extent, axis))
    if not candidates:
        raise ValueError(
            "PyNTT paged-attention decomposition requires a common physical "
            "block hierarchy axis with extent greater than one."
        )
    extent, axis = min(candidates)
    return axis, extent


def head_divisible_paged_attention_split_plan(placements, query_heads):
    """Like ``paged_attention_split_plan``, but keep the query heads shardable.

    Split partials cannot hold their query heads on the split axis, so the
    heads can only be sharded over the remaining mesh axes. When the default
    (smallest) axis leaves a head count that the remaining extent does not
    divide, the heads are replicated and every owner computes all of them.
    Return the smallest qualifying axis whose remaining extent divides every
    head count, or ``None`` when the default already does or none does.
    """

    placements = tuple(placements)
    query_heads = tuple(int(heads) for heads in query_heads)
    if not query_heads:
        return None
    default_axis, _ = paged_attention_split_plan(placements)
    hierarchy = placements[0].hierarchy

    def divides(axis):
        remaining = 1
        for other, extent in enumerate(hierarchy):
            if other != axis:
                remaining *= extent
        return all(heads % remaining == 0 for heads in query_heads)

    if divides(default_axis):
        return None
    candidates = sorted(
        (placements[0].hierarchy[axis], axis)
        for axis in range(placements[0].rank)
        if all(placement.is_physical_block_axis(axis) for placement in placements)
        and len({placement.hierarchy[axis] for placement in placements}) == 1
        and placements[0].hierarchy[axis] > 1
    )
    for extent, axis in candidates:
        if divides(axis):
            return axis, extent
    return None


def decompose_paged_attention_rule(
    split_hierarchy_axis: int,
    split_count: int,
) -> RewriteRule:
    """Use nncase's split-state decomposition for causal query chunks.

    Partial/combine preserves the sequence axis and computes a separate causal
    prefix for each row; the original single-token restriction is unnecessary.
    """

    if split_hierarchy_axis < 0:
        raise ValueError("split_hierarchy_axis must be non-negative.")
    if split_count <= 1:
        raise ValueError("split_count must be greater than one.")
    pattern = F.nn.is_paged_attention(
        wildcard("q"),
        wildcard("state"),
        wildcard("layer_id"),
        target_name="target",
        call_name="call",
    )

    def rewrite(result, module: IRModule):
        source = result["call"]
        q = result["q"]
        state = result["state"]
        layer_id = result["layer_id"]
        assert all(
            isinstance(value, Node) for value in (source, q, state, layer_id)
        )
        layout = tuple(source.attrs["layout"])
        seq_axis = layout.index("seq")
        shape = tensor_of(source.type).shape
        if shape[seq_axis].is_fixed and shape[seq_axis].fixed_value == 0:
            return source
        metadata = decomposition_metadata(source, "DecomposePagedAttention")
        partial = make_node(
            PagedAttentionPartial.op_name,
            f"{source.id}.partial",
            (q, state, layer_id),
            {
                "scale": source.attrs["scale"],
                "layout": layout,
                "hidden_size": source.attrs["hidden_size"],
                "split_hierarchy_axis": split_hierarchy_axis,
                "split_count": split_count,
                **({"probability_rounding": source.attrs["probability_rounding"],
                    "softmax_block_size": source.attrs["softmax_block_size"],
                    "softmax_traversal": source.attrs.get("softmax_traversal", "forward")}
                   if source.attrs.get("probability_rounding", "normalized") != "normalized" else {}),
            },
            metadata,
        )
        projections = tuple(
            make_node(
                "builtin.get_item",
                f"{source.id}.{name}_state",
                (partial,),
                {"index": index},
                {
                    "decomposed_from": source.id,
                    "decomposition_rule": "DecomposePagedAttention",
                },
            )
            for index, name in enumerate(("max", "sum", "acc"))
        )
        replacement = make_node(
            "ntt.paged_attention_combine",
            source.id,
            projections,
            {
                "layout": layout,
                "hidden_size": source.attrs["hidden_size"],
                "output_data_type": tensor_of(source.type).dtype,
                "output_type": source.type,
                "split_hierarchy_axis": split_hierarchy_axis,
                "split_count": split_count,
            },
            metadata,
        )
        return RewriteResult(replacement, (partial, *projections))

    return RewriteRule(
        "DecomposePagedAttention",
        pattern,
        rewrite,
        effect_policy=RewriteEffectPolicy.ALLOW,
    )


__all__ = [
    "decompose_paged_attention_rule",
    "head_divisible_paged_attention_split_plan",
    "paged_attention_split_plan",
]
