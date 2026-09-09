# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Propagate a final-axis Pack through RoPE."""

from __future__ import annotations

from collections.abc import Mapping

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import IRModule, Node, TensorType, get_definition
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.tensors.pack import normalize_axes
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.ntt.vectorize.utility import (
    propagation_helper_metadata,
    propagation_result_metadata,
)


def _make(
    node_id: str,
    op: str,
    inputs: tuple[Node, ...],
    attrs: Mapping[str, object],
    metadata: Mapping[str, object],
) -> Node:
    prepared = get_definition(op).prepare(inputs, attrs)
    return Node(
        node_id,
        op,
        tuple(value.id for value in prepared.inputs),
        prepared.result_type,
        effect=prepared.effect,
        attrs=prepared.attrs,
        metadata=metadata,
    )


def _plan(node: Node, module: IRModule):
    if node.op != "tensors.pack" or len(node.inputs) != 1:
        return None
    rope = module.node_map[node.inputs[0]]
    if rope.op != "nn.rope" or len(rope.inputs) != 3:
        return None
    if not isinstance(rope.type, TensorType):
        return None
    lanes = tuple(int(value) for value in node.attrs["lanes"])
    raw_axes = (
        tuple(int(value) for value in node.attrs["axes"])
        if "axes" in node.attrs
        else (int(node.attrs["axis"]),) * len(lanes)
    )
    axes = normalize_axes(raw_axes, rope.type.rank)
    if len(axes) != 1 or len(lanes) != 1 or axes[0] != rope.type.rank - 1:
        return None
    lane = lanes[0]
    head_dim = rope.type.shape[-1]
    if not head_dim.is_fixed or head_dim.fixed_value % lane:
        return None
    cos = module.node_map[rope.inputs[1]]
    sin = module.node_map[rope.inputs[2]]
    for table in (cos, sin):
        table_type = tensor_of(table.type)
        table_axis = axes[0] - (rope.type.rank - table_type.rank)
        if table_axis < 0 or table_axis >= table_type.rank:
            return None
        table_extent = table_type.shape[table_axis]
        if not table_extent.is_fixed or table_extent.fixed_value % (2 * lane):
            return None
    return rope, axes, lanes


def _matches(node: Node, module: IRModule) -> bool:
    try:
        return _plan(node, module) is not None
    except (IRSchemaError, KeyError, TypeError, ValueError):
        return False


def _rewrite(node: Node, module: IRModule) -> RewriteResult:
    plan = _plan(node, module)
    assert plan is not None
    rope, axes, lanes = plan
    lane = lanes[0]
    input_value = module.node_map[rope.inputs[0]]
    input_pack = _make(
        f"{node.id}.propagated.input_pack",
        "tensors.pack",
        (input_value,),
        {"lanes": lanes, "axes": axes},
        propagation_helper_metadata(node, node.id, "propagated-pack"),
    )
    helpers: list[Node] = [input_pack]
    packed_tables: list[Node] = []
    for name, input_id in (("cos", rope.inputs[1]), ("sin", rope.inputs[2])):
        table = module.node_map[input_id]
        table_type = tensor_of(table.type)
        table_axis = axes[0] - (rope.type.rank - table_type.rank)
        packed = _make(
            f"{node.id}.propagated.{name}_pack",
            "tensors.pack",
            (table,),
            {"lanes": (2, lane), "axes": (table_axis, table_axis)},
            propagation_helper_metadata(node, node.id, "propagated-pack"),
        )
        helpers.append(packed)
        packed_tables.append(packed)
    replacement = _make(
        node.id,
        "ntt.vectorized_rope",
        (input_pack, *packed_tables),
        rope.attrs,
        propagation_result_metadata(
            rope,
            node,
            axes=axes,
            lanes=lanes,
            rule="VectorizeRoPEPropagation",
            internal_role="propagated-compute",
        ),
    )
    return RewriteResult(replacement, tuple(helpers))


def rope_propagation_rules() -> tuple[RewriteRule, ...]:
    return (RewriteRule("VectorizeRoPEPropagation", _matches, _rewrite),)


__all__ = ["rope_propagation_rules"]
