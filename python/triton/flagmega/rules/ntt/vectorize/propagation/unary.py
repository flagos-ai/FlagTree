# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Push Pack/Unpack through unary dataflow."""

from __future__ import annotations

from triton.flagmega.ir import IRModule, Node, TensorType, VectorType
from triton.flagmega.ir.ops.tensors.pack import Pack
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.ntt.vectorize.utility import (
    propagation_helper_metadata,
    propagation_result_metadata,
)


def _pack_unary_matches(node: Node, module: IRModule) -> bool:
    return node.op == "tensors.pack" and module.node_map[node.inputs[0]].op == "math.silu"


def _pack_unary(node: Node, module: IRModule) -> RewriteResult:
    scalar = module.node_map[node.inputs[0]]
    source = module.node_map[scalar.inputs[0]]
    attrs = dict(node.attrs)
    lanes = tuple(int(value) for value in attrs["lanes"])
    axes = (
        tuple(int(value) for value in attrs["axes"])
        if "axes" in attrs
        else (int(attrs["axis"]),) * len(lanes)
    )
    packed_type = Pack.infer_type((source,), attrs)
    pack = Node(
        f"{node.id}.propagated.pack",
        "tensors.pack",
        (source.id,),
        packed_type,
        attrs=attrs,
        metadata=propagation_helper_metadata(node, node.id, "propagated-pack"),
    )
    replacement = Node(
        node.id,
        "math.vectorized_unary",
        (pack.id,),
        node.type,
        attrs={"unary_op": "silu"},
        metadata=propagation_result_metadata(
            scalar,
            node,
            axes=axes,
            lanes=lanes,
            rule="VectorizeUnaryPropagation",
            internal_role="propagated-compute",
        ),
    )
    return RewriteResult(replacement, (pack,))


def _unary_unpack_matches(node: Node, module: IRModule) -> bool:
    if node.op != "math.silu":
        return False
    source = module.node_map[node.inputs[0]]
    return source.op == "tensors.unpack"


def _unary_unpack(node: Node, module: IRModule) -> RewriteResult:
    unpack = module.node_map[node.inputs[0]]
    vector = module.node_map[unpack.inputs[0]]
    assert isinstance(vector.type, TensorType) and isinstance(vector.type.dtype, VectorType)
    axes = (
        tuple(int(value) for value in unpack.attrs["axes"])
        if "axes" in unpack.attrs
        else (int(unpack.attrs["axis"]),) * len(vector.type.dtype.lanes)
    )
    compute = Node(
        f"{node.id}.propagated.compute",
        "math.vectorized_unary",
        (vector.id,),
        vector.type,
        attrs={"unary_op": "silu"},
        metadata=propagation_result_metadata(
            node,
            unpack,
            axes=axes,
            lanes=vector.type.dtype.lanes,
            rule="UnaryDevectorizePropagation",
            internal_role="propagated-compute",
        ),
    )
    replacement = Node(
        node.id,
        "tensors.unpack",
        (compute.id,),
        node.type,
        attrs=dict(unpack.attrs),
        metadata=propagation_result_metadata(
            node,
            unpack,
            axes=axes,
            lanes=vector.type.dtype.lanes,
            rule="UnaryDevectorizePropagation",
        ),
    )
    return RewriteResult(replacement, (compute,))


def unary_propagation_rules() -> tuple[RewriteRule, ...]:
    return (
        RewriteRule("VectorizeUnaryPropagation", _pack_unary_matches, _pack_unary),
        RewriteRule("UnaryDevectorizePropagation", _unary_unpack_matches, _unary_unpack),
    )


__all__ = ["unary_propagation_rules"]
