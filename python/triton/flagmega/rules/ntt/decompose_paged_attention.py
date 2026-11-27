# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Expose split-KV paged-attention states before distributed search."""

from __future__ import annotations

import math
from dataclasses import dataclass

from triton.flagmega.ir import IRModule, Node
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.nn._paged_attention_state import (
    paged_attention_state_config_from_type, )
from triton.flagmega.ir.ops.ntt.paged_attention_partial import (
    PagedAttentionPartial, )
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


@dataclass(frozen=True)
class PagedAttentionWork:
    """The attention workload facts a context-split plan must consult.

    ``num_query_heads``/``num_kv_heads`` carry the grouped-query (GQA) group,
    and ``max_context_length``/``cache_block_size`` carry the cache geometry.
    The decode-time context length is a runtime value (``slot_mapping``), so
    the cache's maximum context is the static work bound available to a
    compile-time plan; the plan records which bound it used.
    """

    num_query_heads: int
    num_kv_heads: int
    max_context_length: int
    cache_block_size: int

    def __post_init__(self) -> None:
        values = (
            self.num_query_heads,
            self.num_kv_heads,
            self.max_context_length,
            self.cache_block_size,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError("PagedAttentionWork requires positive integer head and cache "
                             f"extents, got {values!r}.")
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError(f"PagedAttentionWork query heads {self.num_query_heads} are not "
                             f"a whole multiple of KV heads {self.num_kv_heads}.")
        if self.max_context_length % self.cache_block_size:
            raise ValueError(f"PagedAttentionWork context {self.max_context_length} is not a "
                             f"whole multiple of cache block {self.cache_block_size}.")

    @property
    def query_heads_per_kv_head(self) -> int:
        """The GQA group: query heads that share one KV stream."""

        return self.num_query_heads // self.num_kv_heads

    @property
    def context_pages(self) -> int:
        """Cache pages the context can occupy; the coarse context work unit."""

        return self.max_context_length // self.cache_block_size


def paged_attention_work(module: IRModule) -> PagedAttentionWork | None:
    """Read the attention workload facts out of the module's attention ops.

    Returns ``None`` when the module contains no paged attention at all: there
    is then no attention work to plan a split for and nothing to decompose.
    Modules that *do* contain attention must agree on one workload.
    """

    found: set[PagedAttentionWork] = set()
    for node in module.nodes:
        if node.op not in {"nn.paged_attention", PagedAttentionPartial.op_name}:
            continue
        found.add(_work_of_attention_node(node, module))
    if not found:
        return None
    if len(found) != 1:
        raise ValueError("PyNTT paged-attention decomposition requires one attention "
                         f"workload per module, got {sorted(map(repr, found))}.")
    return next(iter(found))


def _work_of_attention_node(node: Node, module: IRModule) -> PagedAttentionWork:
    layout = tuple(str(value) for value in node.attrs["layout"])
    head_axis = layout.index("head")
    query = tensor_of(module.node_map[node.inputs[0]].type)
    heads = query.shape[head_axis]
    if not heads.is_fixed:
        raise ValueError("PyNTT paged-attention decomposition requires a fixed query head "
                         f"extent, got {heads!r}.")
    config = paged_attention_state_config_from_type(module.node_map[node.inputs[1]].type)
    return PagedAttentionWork(
        num_query_heads=int(heads.fixed_value),
        num_kv_heads=int(config.num_kv_heads),
        max_context_length=int(config.max_sequence_length),
        cache_block_size=int(config.block_size),
    )


def paged_attention_split_plan(placements, work: PagedAttentionWork):
    """Choose the context-split hierarchy axis from the attention workload.

    The split count is a distribution decision about *work*, not about mesh
    shape.  Two workload facts order the admissible physical-block axes:

    1.  KV-stream ownership.  The hierarchy axes the split does **not** take
        are what remains to carry the query-head axis.  When that remaining
        extent exceeds ``num_kv_heads`` the same KV stream is owned, fetched
        and issued by several participants; when it is smaller each owner must
        stream several KV heads.  The plan prefers the axis that leaves
        exactly one KV stream per owner, which is the same as owning a whole
        GQA group locally.
    2.  Context work.  Beyond the cache's page count a finer split cannot give
        every owner context to read, so additional shards only add partials to
        the complete reduction.  This term orders candidates of **equal** KV
        ownership; it never overrides fact 1, because a duplicated KV stream
        costs the active participants on every tile they read while an idle
        participant costs only its partial.

    Ties fall back to the lowest axis index, which keeps a single-candidate
    mesh and a KV-symmetric mesh deterministic.
    """

    placements = tuple(placements)
    if not placements:
        raise ValueError("PyNTT paged-attention decomposition requires at least one placement.")
    if not isinstance(work, PagedAttentionWork):
        raise TypeError("PyNTT paged-attention decomposition requires a PagedAttentionWork "
                        f"describing the attention workload, got {work!r}.")
    rank = placements[0].rank
    if any(placement.rank != rank for placement in placements):
        raise ValueError("PyNTT paged-attention decomposition requires equal placement ranks.")
    candidates = _admissible_split_axes(placements, rank)
    if not candidates:
        raise ValueError("PyNTT paged-attention decomposition requires a common physical "
                         "block hierarchy axis with extent greater than one.")
    axis, extent = min(candidates, key=lambda candidate: _split_plan_order(candidate, placements, rank, work))
    return axis, extent


def _admissible_split_axes(placements, rank: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for axis in range(rank):
        if not all(placement.is_physical_block_axis(axis) for placement in placements):
            continue
        extents = {placement.hierarchy[axis] for placement in placements}
        if len(extents) != 1:
            continue
        extent = next(iter(extents))
        if extent > 1:
            candidates.append((axis, extent))
    return candidates


def _split_plan_order(
    candidate: tuple[int, int],
    placements,
    rank: int,
    work: PagedAttentionWork,
) -> tuple[int, int, int, int]:
    axis, extent = candidate
    owners = _remaining_owner_extent(placements, rank, axis)
    kv_heads = work.num_kv_heads
    return (
        # Participants that would own, fetch and issue the same KV stream.
        math.ceil(owners / kv_heads),
        # KV streams one participant would have to fetch.
        math.ceil(kv_heads / owners),
        # Context shards the cache can actually feed; more is better.
        -min(extent, max(work.context_pages, 1)),
        axis,
    )


def _remaining_owner_extent(placements, rank: int, split_axis: int) -> int:
    """Participants left to carry the query-head axis beside the split."""

    result = 1
    for axis in range(rank):
        if axis == split_axis:
            continue
        result *= max(placement.hierarchy[axis] for placement in placements)
    return result


def decompose_paged_attention_rule(
    split_hierarchy_axis: int,
    split_count: int,
) -> RewriteRule:
    """Use the reference split-state decomposition for causal query chunks.

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
        assert all(isinstance(value, Node) for value in (source, q, state, layer_id))
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
                "scale":
                source.attrs["scale"],
                "layout":
                layout,
                "hidden_size":
                source.attrs["hidden_size"],
                "split_hierarchy_axis":
                split_hierarchy_axis,
                "split_count":
                split_count,
                **({
                    "probability_rounding": source.attrs["probability_rounding"],
                    "softmax_block_size": source.attrs["softmax_block_size"], "softmax_traversal": source.attrs.get(
                        "softmax_traversal", "forward")
                } if source.attrs.get("probability_rounding", "normalized") != "normalized" else {}),
            },
            metadata,
        )
        projections = tuple(
            make_node(
                "builtin.get_item",
                f"{source.id}.{name}_state",
                (partial, ),
                {"index": index},
                {
                    "decomposed_from": source.id,
                    "decomposition_rule": "DecomposePagedAttention",
                },
            ) for index, name in enumerate(("max", "sum", "acc")))
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
    "PagedAttentionWork",
    "decompose_paged_attention_rule",
    "paged_attention_split_plan",
    "paged_attention_work",
]
