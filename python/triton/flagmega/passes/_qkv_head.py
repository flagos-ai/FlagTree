# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Typed Q/K normalization and rotation boundaries for semantic region formation.

Matching records original nodes, never fabricated nodes with inconsistent types.
The wide form removes lossless promotions only; projection/table BF16 stores and
the final BF16 store remain part of the fused operation's numerical contract.
"""

from dataclasses import dataclass

from triton.flagmega.ir import DType, Node, TensorType
from triton.flagmega.ir.ops.nn._norm import normalize_axis


@dataclass(frozen=True)
class QKVHead:
    norm: Node
    rope: Node
    value: Node
    cosine: Node
    sine: Node
    round_intermediates: bool


def cast_source(node, nodes, source_dtype, target_dtype):
    if (node.op != "tensors.cast" or not isinstance(node.type, TensorType)
            or node.type.dtype != target_dtype):
        return None
    source = nodes[node.inputs[0]]
    return source if isinstance(source.type, TensorType) and source.type.dtype == source_dtype else None


def same_table(lhs: Node, rhs: Node) -> bool:
    # Distinct variables with equal type/attrs are NOT equal values. Only a
    # shared node, or the same pure cast applied to that shared node, is proven.
    return lhs.id == rhs.id or (
        lhs.op == rhs.op == "tensors.cast"
        and (lhs.inputs, lhs.attrs, lhs.type) == (rhs.inputs, rhs.attrs, rhs.type)
    )


def match_qkv_head(root: Node, nodes, users) -> QKVHead | None:
    wide = root.op == "tensors.cast"
    rope = cast_source(root, nodes, DType.FLOAT32, DType.BFLOAT16) if wide else root
    if rope is None or rope.op != "nn.rope":
        return None
    norm = nodes[rope.inputs[0]]
    if norm.op != "nn.norm_apply" or not _has_matching_norm_stats(norm, nodes):
        return None
    if users.get(norm.id, ()) != (rope.id,):
        return None
    value = nodes[norm.inputs[0]]
    if (norm.attrs.get("output_dtype") is not None
            and norm.attrs["output_dtype"] != getattr(value.type, "tensor", value.type).dtype.value):
        # This fused QKV contract rounds normalization in the value dtype;
        # it does not encode a distinct normalization-output conversion.
        return None
    tables = tuple(nodes[name] for name in rope.inputs[1:])
    if wide:
        if users.get(rope.id, ()) != (root.id,):
            return None
        value = cast_source(value, nodes, DType.BFLOAT16, DType.FLOAT32)
        tables = tuple(cast_source(table, nodes, DType.BFLOAT16, DType.FLOAT32) for table in tables)
        if value is None or any(table is None for table in tables):
            return None
    return QKVHead(norm, rope, value, *tables, round_intermediates=not wide)


def _has_matching_norm_stats(norm: Node, nodes) -> bool:
    stats = nodes[norm.inputs[1]]
    value = nodes[norm.inputs[0]]
    if (stats.op != "nn.norm_stats" or stats.inputs != (value.id,)
            or bool(stats.attrs.get("use_mean")) != bool(norm.attrs.get("use_mean"))):
        return False
    value_type = value.type.tensor if hasattr(value.type, "tensor") else value.type
    return normalize_axis(int(stats.attrs["axis"]), value_type.rank) == normalize_axis(
        int(norm.attrs["axis"]), value_type.rank)
