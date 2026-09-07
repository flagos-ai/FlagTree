# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Form nncase-compatible semantic QKV/RoPE/cache-update regions."""

from __future__ import annotations

from dataclasses import dataclass, replace

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import (
    IRModule,
    Node,
    PURE,
    TupleType,
    get_definition,
    verify_module,
)
from triton.flagmega.ir.ops.nn._attention_layout import normalize_attention_layout
from triton.flagmega.ir.ops.nn._norm import normalize_axis
from triton.flagmega.ir.ops.nn._paged_attention_state import (
    paged_attention_state_config_from_type,
)


@dataclass(frozen=True)
class _LayoutView:
    root: Node
    source: Node
    input_layout: tuple[str, str, str]
    nodes: tuple[Node, ...]


@dataclass(frozen=True)
class _Fusion:
    paged_attention: Node
    query_view: _LayoutView
    key_view: _LayoutView
    value_view: _LayoutView
    q_norm: Node
    k_norm: Node
    q_rope: Node
    k_rope: Node
    key_update: Node
    value_update: Node


def form_qkv_rope_with_cache(module: IRModule) -> IRModule:
    """Fuse every safe Q/K normalization + RoPE + K/V update region.

    Formation is deliberately target-independent and runs after normalization
    decomposition but before AutoVectorize, matching nncase.  Effectful cache
    updates are removed only when the fused replacement carries the same
    read/write effect and every internal region boundary has a single user.
    """

    current = verify_module(module)
    while True:
        users = _users(current)
        fusion = next(
            (
                match
                for node in current.nodes
                if node.op == "nn.paged_attention"
                and (match := _try_match(node, current, users)) is not None
            ),
            None,
        )
        if fusion is None:
            return current
        current = verify_module(_apply_fusion(current, fusion))


def _try_match(
    paged: Node,
    module: IRModule,
    users: dict[str, tuple[str, ...]],
) -> _Fusion | None:
    node_map = module.node_map
    if len(paged.inputs) != 3:
        return None
    attention_layout = _layout(paged.attrs.get("layout"))
    value_update = node_map[paged.inputs[1]]
    if not _is_cache_update(value_update, "value"):
        return None
    key_update = node_map[value_update.inputs[1]]
    if not _is_cache_update(key_update, "key"):
        return None
    if (
        _layout(key_update.attrs.get("layout")) != attention_layout
        or _layout(value_update.attrs.get("layout")) != attention_layout
        or key_update.inputs[1] == value_update.id
        or key_update.inputs[2] != paged.inputs[2]
        or value_update.inputs[2] != paged.inputs[2]
        or not _is_false_scalar(node_map[key_update.inputs[3]])
    ):
        return None

    try:
        cache_config = paged_attention_state_config_from_type(
            node_map[key_update.inputs[1]].type
        )
    except IRSchemaError:
        return None

    query_view = _match_layout_view(
        node_map[paged.inputs[0]], attention_layout, node_map, cache_config.lanes
    )
    key_view = _match_layout_view(
        node_map[key_update.inputs[0]], attention_layout, node_map, cache_config.lanes
    )
    value_view = _match_layout_view(
        node_map[value_update.inputs[0]], attention_layout, node_map, cache_config.lanes
    )
    if (
        query_view is None
        or key_view is None
        or value_view is None
        or query_view.input_layout != key_view.input_layout
        or query_view.input_layout != value_view.input_layout
    ):
        return None

    q_rope = query_view.source
    k_rope = key_view.source
    if q_rope.op != "nn.rope" or k_rope.op != "nn.rope":
        return None
    if (
        q_rope.inputs[1:] != k_rope.inputs[1:]
        or len(q_rope.inputs) != 3
        or len(k_rope.inputs) != 3
    ):
        return None
    q_norm = node_map[q_rope.inputs[0]]
    k_norm = node_map[k_rope.inputs[0]]
    if q_norm.op != "nn.norm_apply" or k_norm.op != "nn.norm_apply":
        return None
    if not _has_matching_norm_stats(q_norm, node_map) or not _has_matching_norm_stats(
        k_norm, node_map
    ):
        return None
    if (
        not _layout_view_has_only_user(query_view, paged.id, users)
        or not _layout_view_has_only_user(key_view, key_update.id, users)
        or not _layout_view_has_only_user(value_view, value_update.id, users)
        or not _has_only_user(q_rope.id, _source_user(query_view, paged.id), users)
        or not _has_only_user(q_norm.id, q_rope.id, users)
        or not _has_only_user(k_rope.id, _source_user(key_view, key_update.id), users)
        or not _has_only_user(k_norm.id, k_rope.id, users)
        or not _has_only_user(key_update.id, value_update.id, users)
    ):
        return None
    return _Fusion(
        paged,
        query_view,
        key_view,
        value_view,
        q_norm,
        k_norm,
        q_rope,
        k_rope,
        key_update,
        value_update,
    )


def _apply_fusion(module: IRModule, fusion: _Fusion) -> IRModule:
    node_map = module.node_map
    occupied = set(node_map)
    qkv_id = _fresh_id(f"{fusion.value_update.id}.qkv", occupied)
    occupied.add(qkv_id)
    fused_id = _fresh_id(
        f"{fusion.value_update.id}.qkv_rope_with_cache", occupied
    )
    occupied.add(fused_id)
    query_id = _fresh_id(f"{fused_id}.query", occupied)

    qkv_inputs = (
        fusion.q_norm.inputs[0],
        fusion.k_norm.inputs[0],
        fusion.value_view.source.id,
    )
    qkv = Node(
        qkv_id,
        "builtin.tuple",
        qkv_inputs,
        TupleType(tuple(node_map[value].type for value in qkv_inputs)),
        metadata={"formed_by": "FormQKVRoPEWithCache"},
    )
    definition = get_definition("nn.qkv_rope_with_cache")
    attrs = definition.normalize_attrs({
        "q_axis": fusion.q_norm.attrs["axis"],
        "q_epsilon": fusion.q_norm.attrs["epsilon"],
        "q_use_mean": fusion.q_norm.attrs["use_mean"],
        "q_round_before_scale": bool(fusion.q_norm.attrs.get("round_before_scale", False)),
        "k_axis": fusion.k_norm.attrs["axis"],
        "k_epsilon": fusion.k_norm.attrs["epsilon"],
        "k_use_mean": fusion.k_norm.attrs["use_mean"],
        "k_round_before_scale": bool(fusion.k_norm.attrs.get("round_before_scale", False)),
        "qkv_layout": fusion.query_view.input_layout,
        "attention_layout": fusion.paged_attention.attrs["layout"],
    })
    fused_inputs = (
        qkv_id,
        fusion.q_norm.inputs[2],
        fusion.k_norm.inputs[2],
        fusion.q_norm.inputs[3],
        fusion.k_norm.inputs[3],
        fusion.q_rope.inputs[1],
        fusion.q_rope.inputs[2],
        fusion.key_update.inputs[1],
        fusion.paged_attention.inputs[2],
        fusion.value_update.inputs[3],
    )
    input_nodes = tuple(qkv if value == qkv_id else node_map[value] for value in fused_inputs)
    fused_type = definition.infer_type(input_nodes, attrs)
    fused = Node(
        fused_id,
        "nn.qkv_rope_with_cache",
        fused_inputs,
        fused_type,
        definition.infer_effect(input_nodes, attrs),
        attrs,
        {
            **dict(fusion.value_update.metadata),
            "formed_by": "FormQKVRoPEWithCache",
        },
    )
    assert isinstance(fused_type, TupleType) and len(fused_type.fields) == 2
    query = Node(
        query_id,
        "builtin.get_item",
        (fused_id,),
        fused_type.fields[0],
        PURE,
        {"index": 0},
        dict(fusion.query_view.root.metadata),
    )
    state = Node(
        fusion.value_update.id,
        "builtin.get_item",
        (fused_id,),
        fused_type.fields[1],
        PURE,
        {"index": 1},
        dict(fusion.value_update.metadata),
    )

    rewritten: list[Node] = []
    for node in module.nodes:
        if node.id == fusion.key_update.id:
            continue
        if node.id == fusion.value_update.id:
            rewritten.extend((qkv, fused, query, state))
            continue
        rewritten.append(replace(
            node,
            inputs=tuple(
                query_id if value == fusion.query_view.root.id else value
                for value in node.inputs
            ),
        ))
    functions = tuple(replace(
        function,
        outputs=tuple(
            query_id if value == fusion.query_view.root.id else value
            for value in function.outputs
        ),
    ) for function in module.functions)
    return _remove_unused(replace(module, nodes=tuple(rewritten), functions=functions))


def _fresh_id(base: str, occupied: set[str]) -> str:
    if base not in occupied:
        return base
    ordinal = 1
    while f"{base}_{ordinal}" in occupied:
        ordinal += 1
    return f"{base}_{ordinal}"


def _match_layout_view(
    root: Node,
    output_layout: tuple[str, str, str],
    node_map,
    lane: int,
) -> _LayoutView | None:
    current = root
    nodes: list[Node] = []
    expected_axes = (output_layout.index("dim"),)
    if (
        current.op != "tensors.pack"
        or tuple(current.attrs.get("lanes", ())) != (lane,)
        or tuple(current.attrs.get("axes", ())) != expected_axes
    ):
        return None
    nodes.append(current)
    current = node_map[current.inputs[0]]
    input_layout = output_layout
    if current.op == "tensors.permute":
        axes = tuple(int(value) for value in current.attrs.get("axes", ()))
        if sorted(axes) != [0, 1, 2]:
            return None
        restored = [""] * 3
        for output_axis, input_axis in enumerate(axes):
            restored[input_axis] = output_layout[output_axis]
        input_layout = tuple(restored)
        nodes.append(current)
        current = node_map[current.inputs[0]]
    return _LayoutView(root, current, input_layout, tuple(nodes))


def _has_matching_norm_stats(norm: Node, node_map) -> bool:
    if len(norm.inputs) != 4:
        return False
    stats = node_map[norm.inputs[1]]
    value = node_map[norm.inputs[0]]
    if (
        stats.op != "nn.norm_stats"
        or stats.inputs != (value.id,)
        or bool(stats.attrs.get("use_mean")) != bool(norm.attrs.get("use_mean"))
    ):
        return False
    try:
        return normalize_axis(int(stats.attrs["axis"]), _rank(value)) == normalize_axis(
            int(norm.attrs["axis"]), _rank(value)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _rank(node: Node) -> int:
    value_type = node.type.tensor if hasattr(node.type, "tensor") else node.type
    return value_type.rank


def _layout(value) -> tuple[str, str, str] | None:
    try:
        return normalize_attention_layout(value)
    except (IRSchemaError, TypeError, ValueError):
        return None


def _is_cache_update(node: Node, kind: str) -> bool:
    return (
        node.op == "nn.update_paged_attention_kv_cache"
        and len(node.inputs) == 4
        and str(node.attrs.get("cache_kind")) == kind
    )


def _is_false_scalar(node: Node) -> bool:
    return node.op in {"builtin.scalar_const", "tir.scalar_const"} and node.attrs.get(
        "value"
    ) is False


def _has_only_user(value: str, expected: str, users: dict[str, tuple[str, ...]]) -> bool:
    return users.get(value, ()) == (expected,)


def _layout_view_has_only_user(
    view: _LayoutView,
    terminal_user: str,
    users: dict[str, tuple[str, ...]],
) -> bool:
    expected = terminal_user
    for node in view.nodes:
        if not _has_only_user(node.id, expected, users):
            return False
        expected = node.id
    return True


def _source_user(view: _LayoutView, terminal_user: str) -> str:
    return terminal_user if not view.nodes else view.nodes[-1].id


def _users(module: IRModule) -> dict[str, tuple[str, ...]]:
    result: dict[str, list[str]] = {node.id: [] for node in module.nodes}
    for node in module.nodes:
        for input_id in node.inputs:
            result[input_id].append(node.id)
    return {key: tuple(value) for key, value in result.items()}


def _remove_unused(module: IRModule) -> IRModule:
    required = {
        node_id
        for function in module.functions
        for node_id in (*function.parameters, *function.outputs)
    }
    required.update(node.id for node in module.nodes if not node.effect.is_pure)
    pending = list(required)
    node_map = module.node_map
    while pending:
        node_id = pending.pop()
        for input_id in node_map[node_id].inputs:
            if input_id not in required:
                required.add(input_id)
                pending.append(input_id)
    return replace(
        module,
        nodes=tuple(node for node in module.nodes if node.id in required),
        selection_points=tuple(
            point
            for point in module.selection_points
            if point.owner is None or point.owner in required
        ),
        selections=tuple(
            selection
            for selection in module.selections
            if any(
                point.id == selection.point_id
                for point in module.selection_points
                if point.owner is None or point.owner in required
            )
        ),
    )


__all__ = ["form_qkv_rope_with_cache"]
