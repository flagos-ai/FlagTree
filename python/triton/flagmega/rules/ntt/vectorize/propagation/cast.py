# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Propagate typed-vector boundaries through element-width-changing casts."""

from __future__ import annotations

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import DType, IRModule, Node, TensorType, VectorType, get_definition
from triton.flagmega.ir.ops.tensors.pack import normalize_axes
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.ntt.vectorize.utility import (
    propagation_helper_metadata,
    propagation_result_metadata,
)


def _make(node_id: str, op: str, inputs: tuple[Node, ...], attrs, metadata) -> Node:
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


def _pack_contract(node: Node, rank: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    lanes = tuple(int(value) for value in node.attrs["lanes"])
    raw_axes = (
        tuple(int(value) for value in node.attrs["axes"])
        if "axes" in node.attrs
        else (int(node.attrs["axis"]),) * len(lanes)
    )
    return normalize_axes(raw_axes, rank), lanes


def _scaled_lanes(
    lanes: tuple[int, ...], numerator_bytes: int, denominator_bytes: int
) -> tuple[int, ...] | None:
    result: list[int] = []
    for lane in lanes:
        numerator = lane * numerator_bytes
        if numerator % denominator_bytes:
            return None
        scaled = numerator // denominator_bytes
        if scaled <= 0:
            return None
        result.append(scaled)
    return tuple(result)


def _pack_cast_plan(node: Node, module: IRModule):
    if node.op != "tensors.pack":
        return None
    cast = module.node_map[node.inputs[0]]
    if cast.op != "tensors.cast":
        return None
    source = module.node_map[cast.inputs[0]]
    if (
        not isinstance(source.type, TensorType)
        or not isinstance(source.type.dtype, DType)
        or not isinstance(cast.type, TensorType)
        or not isinstance(cast.type.dtype, DType)
    ):
        return None
    axes, output_lanes = _pack_contract(node, cast.type.rank)
    input_lanes = _scaled_lanes(
        output_lanes, cast.type.dtype.itemsize, source.type.dtype.itemsize
    )
    if input_lanes is None:
        return None
    try:
        packed_input = get_definition("tensors.pack").prepare(
            (source,), {"axes": axes, "lanes": input_lanes}
        )
    except (IRSchemaError, TypeError, ValueError):
        return None
    return cast, source, axes, input_lanes, output_lanes, packed_input


def _pack_cast_matches(node: Node, module: IRModule) -> bool:
    return _pack_cast_plan(node, module) is not None


def _pack_cast(node: Node, module: IRModule) -> RewriteResult:
    plan = _pack_cast_plan(node, module)
    assert plan is not None
    cast, source, axes, input_lanes, output_lanes, _ = plan
    packed = _make(
        f"{node.id}.propagated.pack",
        "tensors.pack",
        (source,),
        {"axes": axes, "lanes": input_lanes},
        propagation_helper_metadata(node, node.id, "propagated-pack"),
    )
    replacement = _make(
        node.id,
        "ntt.vectorized_cast",
        (packed,),
        {"new_type": VectorType(cast.type.dtype, output_lanes), "vectorize_axes": axes},
        propagation_result_metadata(
            cast,
            node,
            axes=axes,
            lanes=output_lanes,
            rule="VectorizeCastPropagation",
            internal_role="propagated-cast",
        ),
    )
    return RewriteResult(replacement, (packed,))


def _cast_unpack_plan(node: Node, module: IRModule):
    if node.op != "tensors.cast":
        return None
    unpack = module.node_map[node.inputs[0]]
    if unpack.op != "tensors.unpack":
        return None
    vector = module.node_map[unpack.inputs[0]]
    if (
        not isinstance(vector.type, TensorType)
        or not isinstance(vector.type.dtype, VectorType)
        or not isinstance(node.type, TensorType)
        or not isinstance(node.type.dtype, DType)
    ):
        return None
    axis_count = (
        len(tuple(unpack.attrs["axes"]))
        if "axes" in unpack.attrs
        else len(vector.type.dtype.lanes)
    )
    raw_axes = (
        tuple(int(value) for value in unpack.attrs["axes"])
        if "axes" in unpack.attrs
        else (int(unpack.attrs["axis"]),) * axis_count
    )
    axes = normalize_axes(raw_axes, vector.type.rank)
    if len(axes) != len(vector.type.dtype.lanes):
        return None
    output_lanes = _scaled_lanes(
        vector.type.dtype.lanes,
        vector.type.dtype.elem_type.itemsize,
        node.type.dtype.itemsize,
    )
    if output_lanes is None:
        return None
    try:
        prepared = get_definition("ntt.vectorized_cast").prepare(
            (vector,),
            {"new_type": VectorType(node.type.dtype, output_lanes), "vectorize_axes": axes},
        )
    except (IRSchemaError, TypeError, ValueError):
        return None
    return unpack, vector, axes, output_lanes, prepared


def _cast_unpack_matches(node: Node, module: IRModule) -> bool:
    return _cast_unpack_plan(node, module) is not None


def _cast_unpack(node: Node, module: IRModule) -> RewriteResult:
    plan = _cast_unpack_plan(node, module)
    assert plan is not None
    unpack, vector, axes, output_lanes, _ = plan
    compute = _make(
        f"{node.id}.propagated.cast",
        "ntt.vectorized_cast",
        (vector,),
        {"new_type": VectorType(node.type.dtype, output_lanes), "vectorize_axes": axes},
        propagation_result_metadata(
            node,
            unpack,
            axes=axes,
            lanes=output_lanes,
            rule="CastDevectorizePropagation",
            internal_role="propagated-cast",
        ),
    )
    replacement = _make(
        node.id,
        "tensors.unpack",
        (compute,),
        {"axes": axes},
        propagation_result_metadata(
            node,
            unpack,
            axes=axes,
            lanes=output_lanes,
            rule="CastDevectorizePropagation",
        ),
    )
    return RewriteResult(replacement, (compute,))


def _fold_nop_matches(node: Node, module: IRModule) -> bool:
    if node.op != "ntt.vectorized_cast":
        return False
    source = module.node_map[node.inputs[0]]
    return source.type == node.type


def _fold_nop(node: Node, module: IRModule) -> RewriteResult:
    source = module.node_map[node.inputs[0]]
    return RewriteResult(
        Node(
            node.id,
            source.op,
            source.inputs,
            source.type,
            source.effect,
            source.attrs,
            {**dict(source.metadata), **dict(node.metadata)},
        )
    )


def cast_propagation_rules() -> tuple[RewriteRule, ...]:
    return (
        RewriteRule("VectorizeCastPropagation", _pack_cast_matches, _pack_cast),
        RewriteRule("CastDevectorizePropagation", _cast_unpack_matches, _cast_unpack),
        RewriteRule("FoldNopVectorizedCast", _fold_nop_matches, _fold_nop),
    )


__all__ = ["cast_propagation_rules"]
