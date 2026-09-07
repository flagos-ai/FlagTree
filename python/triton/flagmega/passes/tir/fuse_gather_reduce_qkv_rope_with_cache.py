# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fuse private partial-QKV materialization into QKV RoPE/cache TIR."""

from __future__ import annotations

from dataclasses import dataclass, replace

from triton.flagmega.ir import IRModule, Node, TupleType
from triton.flagmega.ir.ops.ntt.gather_reduce_qkv_rope_with_cache import (
    GatherReduceQKVRoPEWithCache,
)
from triton.flagmega.ir.ops.ntt.packed_qkv_parallel_linear_combine import (
    can_materialize_packed_qkv,
)


_VIEW_OPS = frozenset({
    "builtin.identity",
    "distributed.sharded_view",
    "tensors.reshape",
})
_COMBINE_OPS = frozenset({
    "distributed.boxing",
    "ntt.packed_qkv_parallel_linear_combine",
})


@dataclass(frozen=True)
class _FusionMatch:
    consumer: str
    source: str
    materialized_type: TupleType
    logical_type: TupleType
    removed: frozenset[str]


def fuse_gather_reduce_qkv_rope_with_cache(module: IRModule) -> IRModule:
    """Remove a three-field Sum-partial combine used only by QKV RoPE/cache.

    FlagMega graph TIR retains zero-copy views until bufferization, whereas
    nncase's equivalent pass sees those views on TIR ``Buffer`` objects.  This
    pass proves the same single-physical-use condition over SSA alias chains,
    then records both the removed materialization and final logical view types
    on a first-class fused operation.
    """

    users = _users(module)
    matches: list[_FusionMatch] = []
    claimed: set[str] = set()
    for node in module.nodes:
        if node.op != "nn.qkv_rope_with_cache":
            continue
        match = _match(node, module, users)
        if match is None or claimed & match.removed:
            continue
        matches.append(match)
        claimed.update(match.removed)
    if not matches:
        return module

    by_consumer = {match.consumer: match for match in matches}
    removed = frozenset().union(*(match.removed for match in matches))
    nodes = []
    for node in module.nodes:
        if node.id in removed:
            continue
        match = by_consumer.get(node.id)
        if match is None:
            nodes.append(node)
            continue
        attrs = {
            **dict(node.attrs),
            "materialized_qkv_type": match.materialized_type,
            "logical_qkv_type": match.logical_type,
        }
        inputs = (match.source, *node.inputs[1:])
        # Run the definition here as a pass-local postcondition instead of
        # relying only on the module verifier after the rewrite.
        inferred = GatherReduceQKVRoPEWithCache.infer_call_type(
            tuple(module.node_map[value] for value in inputs), attrs
        )
        if inferred != node.type:
            raise ValueError(
                f"Fused QKV result type changed for {node.id!r}: "
                f"{node.type!r} -> {inferred!r}."
            )
        nodes.append(replace(
            node,
            op=GatherReduceQKVRoPEWithCache.op_name,
            inputs=inputs,
            attrs=attrs,
            metadata={
                **dict(node.metadata),
                "fused_partial_qkv_materialization": sorted(
                    match.removed - {node.id}
                ),
            },
        ))

    removed_points = {
        point.id
        for point in module.selection_points
        if point.owner in removed
    }
    return replace(
        module,
        nodes=tuple(nodes),
        selection_points=tuple(
            point for point in module.selection_points
            if point.id not in removed_points
        ),
        selections=tuple(
            record for record in module.selections
            if record.point_id not in removed_points
        ),
    )


def _match(
    consumer: Node,
    module: IRModule,
    users: dict[str, tuple[str, ...]],
) -> _FusionMatch | None:
    if len(consumer.inputs) != 10:
        return None
    node_map = module.node_map
    qkv = node_map.get(consumer.inputs[0])
    if (
        qkv is None
        or qkv.op != "builtin.tuple"
        or len(qkv.inputs) != 3
        or users.get(qkv.id) != (consumer.id,)
        or not isinstance(qkv.type, TupleType)
    ):
        return None

    combine_id: str | None = None
    get_items: list[str] = []
    removed = {qkv.id}
    for index, field_id in enumerate(qkv.inputs):
        expected_user = qkv.id
        current = node_map.get(field_id)
        while current is not None and current.op in _VIEW_OPS:
            if len(current.inputs) != 1 or users.get(current.id) != (expected_user,):
                return None
            removed.add(current.id)
            expected_user = current.id
            current = node_map.get(current.inputs[0])
        if (
            current is None
            or current.op != "builtin.get_item"
            or int(current.attrs.get("index", -1)) != index
            or len(current.inputs) != 1
            or users.get(current.id) != (expected_user,)
        ):
            return None
        current_combine = current.inputs[0]
        if combine_id is None:
            combine_id = current_combine
        elif combine_id != current_combine:
            return None
        get_items.append(current.id)
        removed.add(current.id)

    assert combine_id is not None
    combine = node_map.get(combine_id)
    if (
        combine is None
        or combine.op not in _COMBINE_OPS
        or len(combine.inputs) != 1
        or tuple(sorted(users.get(combine.id, ()))) != tuple(sorted(get_items))
        or not isinstance(combine.type, TupleType)
    ):
        return None
    source = node_map.get(combine.inputs[0])
    if (
        source is None
        or not can_materialize_packed_qkv(source.type, combine.type)
    ):
        return None
    attrs = {
        **dict(consumer.attrs),
        "materialized_qkv_type": combine.type,
        "logical_qkv_type": qkv.type,
    }
    try:
        inferred = GatherReduceQKVRoPEWithCache.infer_call_type(
            (source, *(node_map[value] for value in consumer.inputs[1:])), attrs
        )
    except (TypeError, ValueError, KeyError):
        return None
    if inferred != consumer.type:
        return None
    removed.add(combine.id)
    return _FusionMatch(
        consumer.id,
        source.id,
        combine.type,
        qkv.type,
        frozenset(removed),
    )


def _users(module: IRModule) -> dict[str, tuple[str, ...]]:
    result: dict[str, list[str]] = {node.id: [] for node in module.nodes}
    for node in module.nodes:
        for input_id in node.inputs:
            result.setdefault(input_id, []).append(node.id)
    for function in module.functions:
        for output in function.outputs:
            result.setdefault(output, []).append(f"@{function.name}:return")
    return {key: tuple(value) for key, value in result.items()}


@dataclass(frozen=True)
class FuseGatherReduceQKVRoPEWithCachePass:
    name: str = "FuseGatherReduceQKVRoPEWithCache"
    preserves: frozenset[str] = frozenset()

    def run(self, module: IRModule) -> IRModule:
        return fuse_gather_reduce_qkv_rope_with_cache(module)


__all__ = [
    "FuseGatherReduceQKVRoPEWithCachePass",
    "fuse_gather_reduce_qkv_rope_with_cache",
]
