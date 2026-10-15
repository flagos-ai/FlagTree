# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Form a complete cache publication region before physical layout selection."""

from dataclasses import replace

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import Node, get_definition
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.tensors.pack import _pack_axes
from triton.flagmega.ir.ops.tensors.slice import Slice
from triton.flagmega.pattern_match import F, wildcard
from triton.flagmega.rules.core import RewriteEffectPolicy, RewriteResult, RewriteRule


def form_paged_attention_kv_cache_pair_rule():
    layer = wildcard("layer")
    key = F.nn.is_update_paged_attention_kv_cache(
        wildcard("key"), wildcard("state"), layer, wildcard("no_advance"),
        cache_kind="key", call_name="key_update").with_user_count(1)
    return RewriteRule(
        "form_paged_attention_kv_cache_pair",
        F.nn.is_update_paged_attention_kv_cache(
            wildcard("value"), key, layer, wildcard("advance"),
            cache_kind="value", call_name="value_update"),
        _rewrite, effect_policy=RewriteEffectPolicy.ALLOW,
    )


def _slice_branch(node, module, dim_axis):
    pack = node if node.op == "tensors.pack" else None
    if pack is not None:
        if any(axis != dim_axis for axis in _pack_axes(pack.attrs, len(pack.attrs["lanes"]), 3)):
            return None
        node = module.node_map[pack.inputs[0]]
    if node.op != "tensors.slice":
        return None
    return node, pack


def _rewrite(match, module):
    root, key_update = match["value_update"], match["key_update"]
    layout = tuple(root.attrs["layout"])
    if tuple(key_update.attrs["layout"]) != layout:
        return root
    no_advance = match["no_advance"]
    if no_advance.op not in {"builtin.scalar_const", "tir.scalar_const"} or no_advance.attrs.get("value") is not False:
        return root
    dim_axis = layout.index("dim")
    branches = tuple(_slice_branch(match[name], module, dim_axis) for name in ("key", "value"))
    if any(branch is None for branch in branches):
        return root
    (key, key_pack), (value, value_pack) = branches
    if key.inputs != value.inputs or (key_pack is None) != (value_pack is None):
        return root
    if key_pack is not None and tuple(key_pack.attrs["lanes"]) != tuple(value_pack.attrs["lanes"]):
        return root
    source = module.node_map[key.inputs[0]]
    tensor = tensor_of(source.type)
    if tensor.rank != 3 or not tensor.shape[dim_axis].is_fixed:
        return root
    extent = tensor.shape[dim_axis].fixed_value
    if extent % 2:
        return root
    for sliced, expected in ((key, range(extent // 2)), (value, range(extent // 2, extent))):
        ranges = dict(Slice.ranges(tensor, sliced.attrs))
        if ranges.get(dim_axis) != expected or ranges[dim_axis].step != 1 or any(
                axis != dim_axis and indices != range(tensor.shape[axis].fixed_value)
                for axis, indices in ranges.items()):
            return root
    positions = {node.id: index for index, node in enumerate(module.nodes)}
    # Publishing at the second write is legal only when the original first
    # write has no observer, no other user, and no intervening state effect.
    if any(not node.effect.is_pure for node in module.nodes[positions[key_update.id] + 1:positions[root.id]]):
        return root
    prefix = ()
    if key_pack is not None:
        definition = get_definition("tensors.pack")
        prepared = definition.prepare((source,), key_pack.attrs)
        identifier = root.id + ".pair_slots"
        while identifier in module.node_map:
            identifier += "_"
        source = Node(identifier, "tensors.pack", (source.id,), prepared.result_type,
                      prepared.effect, prepared.attrs, {"formed_by": "FormPagedAttentionKVCachePair"})
        prefix = (source,)
    inputs = (source, match["state"], match["layer"], match["advance"])
    definition = get_definition("nn.update_paged_attention_kv_cache_pair")
    try:
        prepared = definition.prepare(inputs, {"layout": layout})
    except IRSchemaError:
        return root
    replacement = replace(root, op=definition.op_name, inputs=tuple(node.id for node in inputs),
                          type=prepared.result_type, effect=prepared.effect, attrs=prepared.attrs,
                          metadata={**root.metadata, "formed_by": "FormPagedAttentionKVCachePair"})
    return RewriteResult(replacement, prefix, removed_ids=(key_update.id,))


__all__ = ["form_paged_attention_kv_cache_pair_rule"]
