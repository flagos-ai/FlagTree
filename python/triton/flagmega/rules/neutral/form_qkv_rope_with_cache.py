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
)
from triton.flagmega.ir.ops.nn._attention_layout import normalize_attention_layout
from triton.flagmega.ir.ops.tensors.pack import _pack_axes
from triton.flagmega.rules.neutral._qkv_head import QKVHead, qkv_head_pattern, qkv_head_from_match, same_table
from triton.flagmega.pattern_match import F, is_alt, wildcard
from triton.flagmega.ir.ops.nn._paged_attention_state import (
    paged_attention_state_config_from_type, )

from triton.flagmega.rules.core import RewriteEffectPolicy, RewriteResult, RewriteRule


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
    q_head: QKVHead
    k_head: QKVHead
    key_update: Node
    value_update: Node


def _layout_pattern(source, prefix):
    permuted = F.tensors.is_permute(source, call_name=f"{prefix}_permute").with_user_count(1)
    return F.tensors.is_pack(is_alt(source, permuted), call_name=f"{prefix}_pack").with_user_count(1)


def form_qkv_rope_with_cache_rule() -> RewriteRule:
    """Match the complete Q/K normalization, rotation and K/V state chain."""
    layer = wildcard("layer")
    q = _layout_pattern(qkv_head_pattern("q"), "q")
    k = _layout_pattern(qkv_head_pattern("k"), "k")
    v = _layout_pattern(wildcard("v_root"), "v")
    key_state = F.nn.is_update_paged_attention_kv_cache(
        k,
        wildcard("state"),
        layer,
        wildcard("no_advance"),
        cache_kind="key",
        call_name="key_update",
    ).with_user_count(1)
    value_state = F.nn.is_update_paged_attention_kv_cache(
        v,
        key_state,
        layer,
        wildcard("advance"),
        cache_kind="value",
        call_name="value_update",
    )
    return RewriteRule(
        "form_qkv_rope_with_cache",
        F.nn.is_paged_attention(q, value_state, layer, call_name="paged"),
        _rewrite,
        effect_policy=RewriteEffectPolicy.ALLOW,
    )


def _rewrite(result, module: IRModule):
    paged = result["paged"]
    fusion = _legal_region(result, module)
    return paged if fusion is None else _replacement(module, fusion)


def _legal_region(result, module):
    paged, key_update, value_update = (result[name] for name in ("paged", "key_update", "value_update"))
    attention_layout = _layout(paged.attrs["layout"])
    if (_layout(key_update.attrs["layout"]) != attention_layout
            or _layout(value_update.attrs["layout"]) != attention_layout or not _is_false_scalar(result["no_advance"])):
        return None
    try:
        cache = paged_attention_state_config_from_type(result["state"].type)
    except IRSchemaError:
        return None
    views = tuple(_layout_from_match(result, prefix, attention_layout, cache.lanes) for prefix in ("q", "k", "v"))
    if any(view is None for view in views):
        return None
    query, key, value = views
    if query.input_layout != key.input_layout or query.input_layout != value.input_layout:
        return None
    q_head, k_head = (qkv_head_from_match(result, prefix) for prefix in ("q", "k"))
    if q_head is None or k_head is None:
        return None
    if (q_head.round_intermediates != k_head.round_intermediates
            or q_head.rope.attrs.get("rotary_dim") != k_head.rope.attrs.get("rotary_dim")
            or not same_table(q_head.cosine, k_head.cosine) or not same_table(q_head.sine, k_head.sine)):
        return None
    # Pattern establishes sharing and the state chain, not ordering of other
    # effects on the same reference. Do not move a key write past an observer.
    positions = {node.id: index for index, node in enumerate(module.nodes)}
    if any(not node.effect.is_pure for node in module.nodes[positions[key_update.id] + 1:positions[value_update.id]]):
        return None
    # The fused query is published at the original value-write position.
    # Its inputs must already exist there, even if Q was originally computed later.
    operands = (q_head.value, q_head.cosine, q_head.sine)
    input_ids = [node.id for node in operands] + list(q_head.norm.inputs[2:])
    if any(positions[value] >= positions[value_update.id] for value in input_ids):
        return None
    return _Fusion(paged, query, key, value, q_head, k_head, key_update, value_update)


def _replacement(module: IRModule, fusion: _Fusion) -> RewriteResult:
    node_map = module.node_map
    occupied = set(node_map)
    qkv_id = _fresh_id(f"{fusion.value_update.id}.qkv", occupied)
    occupied.add(qkv_id)
    fused_id = _fresh_id(f"{fusion.value_update.id}.qkv_rope_with_cache", occupied)
    occupied.add(fused_id)
    query_id = _fresh_id(f"{fused_id}.query", occupied)

    qkv_inputs = (
        fusion.q_head.value.id,
        fusion.k_head.value.id,
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
        "rotary_dim":
        fusion.q_head.rope.attrs.get("rotary_dim"),
        "round_qk_intermediates":
        fusion.q_head.round_intermediates,
        "q_axis":
        fusion.q_head.norm.attrs["axis"],
        "q_epsilon":
        fusion.q_head.norm.attrs["epsilon"],
        "q_use_mean":
        fusion.q_head.norm.attrs["use_mean"],
        "q_round_before_scale":
        bool(fusion.q_head.norm.attrs.get("round_before_scale", False)),
        "k_axis":
        fusion.k_head.norm.attrs["axis"],
        "k_epsilon":
        fusion.k_head.norm.attrs["epsilon"],
        "k_use_mean":
        fusion.k_head.norm.attrs["use_mean"],
        "k_round_before_scale":
        bool(fusion.k_head.norm.attrs.get("round_before_scale", False)),
        "qkv_layout":
        fusion.query_view.input_layout,
        "attention_layout":
        fusion.paged_attention.attrs["layout"],
    })
    fused_inputs = (
        qkv_id,
        fusion.q_head.norm.inputs[2],
        fusion.k_head.norm.inputs[2],
        fusion.q_head.norm.inputs[3],
        fusion.k_head.norm.inputs[3],
        fusion.q_head.cosine.id,
        fusion.q_head.sine.id,
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
        (fused_id, ),
        fused_type.fields[0],
        PURE,
        {"index": 0},
        dict(fusion.query_view.root.metadata),
    )
    state = Node(
        fusion.value_update.id,
        "builtin.get_item",
        (fused_id, ),
        fused_type.fields[1],
        PURE,
        {"index": 1},
        dict(fusion.value_update.metadata),
    )

    paged = replace(
        fusion.paged_attention,
        inputs=(query_id, *fusion.paged_attention.inputs[1:]),
    )
    return RewriteResult(
        paged,
        (qkv, fused, query),
        extra_replacements=(state, ),
        removed_ids=(fusion.key_update.id, ),
        insertion_before=fusion.value_update.id,
    )


def _fresh_id(base: str, occupied: set[str]) -> str:
    if base not in occupied:
        return base
    ordinal = 1
    while f"{base}_{ordinal}" in occupied:
        ordinal += 1
    return f"{base}_{ordinal}"


def _layout_from_match(result, prefix, output_layout, lane):
    root = result[f"{prefix}_pack"]
    source = result[f"{prefix}_root"]
    if (tuple(root.attrs["lanes"]) != (lane, ) or _pack_axes(root.attrs, 1, 3) != (output_layout.index("dim"), )):
        return None
    permute = result.get_value_or_default(f"{prefix}_permute")
    input_layout = output_layout
    if permute is not None:
        axes = tuple(permute.attrs["axes"])
        if sorted(axes) != [0, 1, 2]:
            return None
        input_layout = tuple(output_layout[axes.index(axis)] for axis in range(3))
    return _LayoutView(root, source, input_layout, (root, ) if permute is None else (root, permute))


def _layout(value) -> tuple[str, str, str] | None:
    try:
        return normalize_attention_layout(value)
    except (IRSchemaError, TypeError, ValueError):
        return None


def _is_false_scalar(node: Node) -> bool:
    return node.op in {"builtin.scalar_const", "tir.scalar_const"} and node.attrs.get("value") is False


__all__ = ["form_qkv_rope_with_cache_rule"]
