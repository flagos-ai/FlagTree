# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Replace a last-axis Unpack view with a storage Bitcast."""

from triton.flagmega.ir import IRModule, Node, VectorType, get_definition
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.tensors.pack import normalize_axes
from triton.flagmega.ir.types import data_type_to_data
from triton.flagmega.pattern_match import F, MatchResult, wildcard
from triton.flagmega.rules import RewriteRule


_INPUT = wildcard("input")
_PATTERN = F.tensors.is_unpack(_INPUT, call_name="unpack")


def _is_last_axis_unpack(node: Node, source: Node) -> bool:
    source_tensor = tensor_of(source.type)
    if not isinstance(source_tensor.dtype, VectorType):
        return False
    count = (
        len(tuple(node.attrs["axes"]))
        if "axes" in node.attrs else len(source_tensor.dtype.lanes)
    )
    axes = normalize_axes(
        tuple(int(value) for value in node.attrs["axes"])
        if "axes" in node.attrs
        else (int(node.attrs["axis"]),) * count,
        source_tensor.rank,
    )
    return bool(axes) and all(axis == source_tensor.rank - 1 for axis in axes)


def _rewrite(result: MatchResult, _module: IRModule) -> Node:
    node = result["unpack"]
    source = result["input"]
    assert isinstance(node, Node) and isinstance(source, Node)
    if not _is_last_axis_unpack(node, source):
        return node
    prepared = get_definition("tensors.bitcast").prepare(
        (source,), {"dtype": data_type_to_data(tensor_of(node.type).dtype)})
    if prepared.result_type != node.type:
        raise ValueError(
            f"UnpackToBitcast inferred {prepared.result_type!r}, expected {node.type!r}.")
    return Node(
        node.id,
        "tensors.bitcast",
        (source.id,),
        prepared.result_type,
        prepared.effect,
        prepared.attrs,
        {**dict(node.metadata), "rewritten_by": "UnpackToBitcast"},
    )


def unpack_to_bitcast_rule() -> RewriteRule:
    return RewriteRule("UnpackToBitcast", pattern=_PATTERN, rewrite=_rewrite)


__all__ = ["unpack_to_bitcast_rule"]
