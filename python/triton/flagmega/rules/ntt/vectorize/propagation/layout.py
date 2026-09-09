# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Propagate typed vector boundaries through layout-preserving tensor ops."""

from __future__ import annotations

from collections.abc import Mapping

from triton.flagmega.ir import (
    IRModule,
    Node,
    TensorType,
    VectorType,
    get_definition,
    try_div_exactly,
)
from triton.flagmega.ir.ops.tensors.pack import normalize_axes
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.ntt.vectorize.utility import (
    propagation_helper_metadata,
    propagation_result_metadata,
)


def _pack_axes(node: Node, rank: int) -> tuple[int, ...]:
    lanes = tuple(int(value) for value in node.attrs["lanes"])
    values = (
        tuple(int(value) for value in node.attrs["axes"])
        if "axes" in node.attrs
        else (int(node.attrs["axis"]),) * len(lanes)
    )
    return normalize_axes(values, rank)


def _unpack_contract(node: Node, module: IRModule) -> tuple[Node, tuple[int, ...], tuple[int, ...]]:
    vector = module.node_map[node.inputs[0]]
    assert isinstance(vector.type, TensorType) and isinstance(vector.type.dtype, VectorType)
    axis_count = len(tuple(node.attrs["axes"])) if "axes" in node.attrs else len(vector.type.dtype.lanes)
    values = (
        tuple(int(value) for value in node.attrs["axes"])
        if "axes" in node.attrs
        else (int(node.attrs["axis"]),) * axis_count
    )
    axes = normalize_axes(values, vector.type.rank)
    return vector, axes, vector.type.dtype.lanes[:len(axes)]


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


def _pack_permute_matches(node: Node, module: IRModule) -> bool:
    return node.op == "tensors.pack" and module.node_map[node.inputs[0]].op == "tensors.permute"


def _pack_permute(node: Node, module: IRModule) -> RewriteResult:
    permute = module.node_map[node.inputs[0]]
    source = module.node_map[permute.inputs[0]]
    axes = _pack_axes(node, permute.type.rank)
    permutation = tuple(int(value) for value in permute.attrs["axes"])
    input_axes = tuple(permutation[axis] for axis in axes)
    packed = _make(
        f"{node.id}.propagated.pack",
        "tensors.pack",
        (source,),
        {"lanes": tuple(node.attrs["lanes"]), "axes": input_axes},
        propagation_helper_metadata(node, node.id, "propagated-pack"),
    )
    replacement = _make(
        node.id,
        "tensors.permute",
        (packed,),
        permute.attrs,
        propagation_result_metadata(
            permute,
            node,
            axes=axes,
            lanes=tuple(int(value) for value in node.attrs["lanes"]),
            rule="VectorizeTransposePropagation",
            internal_role="propagated-permute",
        ),
    )
    return RewriteResult(replacement, (packed,))


def _permute_unpack_matches(node: Node, module: IRModule) -> bool:
    return node.op == "tensors.permute" and module.node_map[node.inputs[0]].op == "tensors.unpack"


def _permute_unpack(node: Node, module: IRModule) -> RewriteResult:
    unpack = module.node_map[node.inputs[0]]
    vector, axes, lanes = _unpack_contract(unpack, module)
    permutation = tuple(int(value) for value in node.attrs["axes"])
    output_axes = tuple(permutation.index(axis) for axis in axes)
    transformed = _make(
        f"{node.id}.propagated.permute",
        "tensors.permute",
        (vector,),
        node.attrs,
        propagation_result_metadata(
            node,
            unpack,
            axes=output_axes,
            lanes=lanes,
            rule="TransposeDevectorizePropagation",
            internal_role="propagated-permute",
        ),
    )
    replacement = _make(
        node.id,
        "tensors.unpack",
        (transformed,),
        {"axes": output_axes},
        propagation_result_metadata(
            node,
            unpack,
            axes=output_axes,
            lanes=lanes,
            rule="TransposeDevectorizePropagation",
        ),
    )
    assert lanes == vector.type.dtype.lanes[:len(output_axes)]
    return RewriteResult(replacement, (transformed,))


def _scaled_shape(
    values: tuple[int, ...],
    axes: tuple[int, ...],
    lanes: tuple[int, ...],
) -> tuple[int, ...] | None:
    result = list(values)
    for axis, lane in zip(axes, lanes):
        if result[axis] % lane:
            return None
        result[axis] //= lane
    return tuple(result)


def _pack_shape_op_matches(node: Node, module: IRModule, op: str, attr: str) -> bool:
    if node.op != "tensors.pack":
        return False
    transform = module.node_map[node.inputs[0]]
    if transform.op != op:
        return False
    axes = _pack_axes(node, transform.type.rank)
    source_type = module.node_map[transform.inputs[0]].type
    if not isinstance(source_type, TensorType) or any(
        try_div_exactly(source_type.shape[axis], lane) is None
        for axis, lane in zip(axes, tuple(int(value) for value in node.attrs["lanes"]))
    ):
        return False
    values = tuple(int(value) for value in transform.attrs[attr])
    return _scaled_shape(values, axes, tuple(int(value) for value in node.attrs["lanes"])) is not None


def _pack_shape_op(node: Node, module: IRModule, op: str, attr: str) -> RewriteResult:
    transform = module.node_map[node.inputs[0]]
    source = module.node_map[transform.inputs[0]]
    axes = _pack_axes(node, transform.type.rank)
    lanes = tuple(int(value) for value in node.attrs["lanes"])
    scaled = _scaled_shape(tuple(int(value) for value in transform.attrs[attr]), axes, lanes)
    assert scaled is not None
    packed = _make(
        f"{node.id}.propagated.pack",
        "tensors.pack",
        (source,),
        {"lanes": lanes, "axes": axes},
        propagation_helper_metadata(node, node.id, "propagated-pack"),
    )
    replacement = _make(
        node.id,
        op,
        (packed,),
        {**dict(transform.attrs), attr: scaled},
        propagation_result_metadata(
            transform,
            node,
            axes=axes,
            lanes=lanes,
            rule=f"Vectorize{op.rsplit('.', 1)[-1].title()}Propagation",
            internal_role=f"propagated-{op.rsplit('.', 1)[-1]}",
        ),
    )
    return RewriteResult(replacement, (packed,))


def _shape_op_unpack_matches(node: Node, module: IRModule, op: str, attr: str) -> bool:
    if node.op != op:
        return False
    unpack = module.node_map[node.inputs[0]]
    if unpack.op != "tensors.unpack":
        return False
    _, axes, lanes = _unpack_contract(unpack, module)
    values = tuple(int(value) for value in node.attrs[attr])
    return _scaled_shape(values, axes, lanes) is not None


def _shape_op_unpack(node: Node, module: IRModule, op: str, attr: str) -> RewriteResult:
    unpack = module.node_map[node.inputs[0]]
    vector, axes, lanes = _unpack_contract(unpack, module)
    scaled = _scaled_shape(tuple(int(value) for value in node.attrs[attr]), axes, lanes)
    assert scaled is not None
    transformed = _make(
        f"{node.id}.propagated.{op.rsplit('.', 1)[-1]}",
        op,
        (vector,),
        {**dict(node.attrs), attr: scaled},
        propagation_result_metadata(
            node,
            unpack,
            axes=axes,
            lanes=lanes,
            rule=f"{op.rsplit('.', 1)[-1].title()}DevectorizePropagation",
            internal_role=f"propagated-{op.rsplit('.', 1)[-1]}",
        ),
    )
    replacement = _make(
        node.id,
        "tensors.unpack",
        (transformed,),
        {"axes": axes},
        propagation_result_metadata(
            node,
            unpack,
            axes=axes,
            lanes=lanes,
            rule=f"{op.rsplit('.', 1)[-1].title()}DevectorizePropagation",
        ),
    )
    return RewriteResult(replacement, (transformed,))


def layout_propagation_rules() -> tuple[RewriteRule, ...]:
    return (
        RewriteRule("VectorizeTransposePropagation", _pack_permute_matches, _pack_permute),
        RewriteRule("TransposeDevectorizePropagation", _permute_unpack_matches, _permute_unpack),
        RewriteRule(
            "VectorizePadPropagation",
            lambda node, module: _pack_shape_op_matches(node, module, "tensors.pad", "pad_end"),
            lambda node, module: _pack_shape_op(node, module, "tensors.pad", "pad_end"),
        ),
        RewriteRule(
            "PadDevectorizePropagation",
            lambda node, module: _shape_op_unpack_matches(node, module, "tensors.pad", "pad_end"),
            lambda node, module: _shape_op_unpack(node, module, "tensors.pad", "pad_end"),
        ),
        RewriteRule(
            "VectorizeSliceToShapePropagation",
            lambda node, module: _pack_shape_op_matches(
                node, module, "tensors.slice_to_shape", "shape"
            ),
            lambda node, module: _pack_shape_op(
                node, module, "tensors.slice_to_shape", "shape"
            ),
        ),
        RewriteRule(
            "SliceToShapeDevectorizePropagation",
            lambda node, module: _shape_op_unpack_matches(
                node, module, "tensors.slice_to_shape", "shape"
            ),
            lambda node, module: _shape_op_unpack(
                node, module, "tensors.slice_to_shape", "shape"
            ),
        ),
    )


__all__ = ["layout_propagation_rules"]
