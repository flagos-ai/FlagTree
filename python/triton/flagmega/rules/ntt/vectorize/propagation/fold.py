# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fold inverse Pack/Unpack representation boundaries."""

from __future__ import annotations

from triton.flagmega.ir import IRModule, Node, TensorType, VectorType
from triton.flagmega.rules import RewriteRedirect, RewriteRule


def _pack_unpack_matches(node: Node, module: IRModule) -> bool:
    if node.op != "tensors.pack":
        return False
    unpack = module.node_map[node.inputs[0]]
    if unpack.op != "tensors.unpack":
        return False
    source = module.node_map[unpack.inputs[0]]
    if not isinstance(source.type, TensorType) or not isinstance(source.type.dtype, VectorType):
        return False
    unpack_axes = _axes(unpack, len(source.type.dtype.lanes))
    pack_axes = _axes(node, len(tuple(node.attrs["lanes"])))
    return (
        tuple(node.attrs["lanes"]) == source.type.dtype.lanes[:len(unpack_axes)]
        and pack_axes == unpack_axes
        and node.type == source.type
    )


def _unpack_pack_matches(node: Node, module: IRModule) -> bool:
    if node.op != "tensors.unpack" or "vectorized_from" in node.metadata:
        return False
    pack = module.node_map[node.inputs[0]]
    if pack.op != "tensors.pack":
        return False
    source = module.node_map[pack.inputs[0]]
    lanes = tuple(pack.attrs["lanes"])
    return _axes(node, len(lanes)) == _axes(pack, len(lanes)) and node.type == source.type


def _axes(node: Node, lane_count: int) -> tuple[int, ...]:
    if "axes" in node.attrs:
        return tuple(node.attrs["axes"])
    return (int(node.attrs["axis"]),) * lane_count


def fold_boundary_rules() -> tuple[RewriteRule, ...]:
    return (
        RewriteRule(
            "FoldPackUnpack",
            _pack_unpack_matches,
            lambda node, module: RewriteRedirect(module.node_map[node.inputs[0]].inputs[0]),
        ),
        RewriteRule(
            "FoldUnpackPack",
            _unpack_pack_matches,
            lambda node, module: RewriteRedirect(module.node_map[node.inputs[0]].inputs[0]),
        ),
    )


__all__ = ["fold_boundary_rules"]
