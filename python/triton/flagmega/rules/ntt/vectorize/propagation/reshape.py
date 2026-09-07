# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Conservative row-major vector propagation through static reshape."""

from __future__ import annotations

from triton.flagmega.ir import IRModule, Node, TensorType, VectorType, get_definition
from triton.flagmega.ir.ops.tensors.pack import normalize_axes
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.ntt.vectorize.utility import (
    propagation_helper_metadata,
    propagation_result_metadata,
)


def _fixed_shape(value: TensorType) -> tuple[int, ...] | None:
    if any(not dimension.is_fixed for dimension in value.shape):
        return None
    return tuple(dimension.fixed_value for dimension in value.shape)


def _strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    result = [1] * len(shape)
    suffix = 1
    for axis in range(len(shape) - 1, -1, -1):
        result[axis] = suffix
        suffix *= shape[axis]
    return tuple(result)


def _map_lane_axes(
    source_shape: tuple[int, ...],
    source_axes: tuple[int, ...],
    lanes: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> tuple[int, ...] | None:
    """Map lane factors only across identical row-major scalar strides."""

    source_strides = _strides(source_shape)
    target_strides = _strides(target_shape)
    products: dict[int, int] = {}
    mapped: list[int] = []
    for source_axis, lane in zip(source_axes, lanes):
        candidates = [
            axis
            for axis, stride in enumerate(target_strides)
            if stride == source_strides[source_axis] and target_shape[axis] != 1
        ]
        selected = next(
            (
                axis
                for axis in reversed(candidates)
                if target_shape[axis] % (products.get(axis, 1) * lane) == 0
            ),
            None,
        )
        if selected is None:
            return None
        products[selected] = products.get(selected, 1) * lane
        mapped.append(selected)
    return tuple(mapped)


def _pack_contract(node: Node, rank: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    lanes = tuple(int(value) for value in node.attrs["lanes"])
    raw_axes = (
        tuple(int(value) for value in node.attrs["axes"])
        if "axes" in node.attrs
        else (int(node.attrs["axis"]),) * len(lanes)
    )
    return normalize_axes(raw_axes, rank), lanes


def _unpack_contract(node: Node, module: IRModule):
    vector = module.node_map[node.inputs[0]]
    if not isinstance(vector.type, TensorType) or not isinstance(vector.type.dtype, VectorType):
        return None
    axis_count = len(tuple(node.attrs["axes"])) if "axes" in node.attrs else len(vector.type.dtype.lanes)
    raw_axes = (
        tuple(int(value) for value in node.attrs["axes"])
        if "axes" in node.attrs
        else (int(node.attrs["axis"]),) * axis_count
    )
    axes = normalize_axes(raw_axes, vector.type.rank)
    return vector, axes, vector.type.dtype.lanes[:len(axes)]


def _make(node_id, op, inputs, attrs, metadata):
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


def _pack_reshape_plan(node: Node, module: IRModule):
    if node.op != "tensors.pack":
        return None
    reshape = module.node_map[node.inputs[0]]
    if reshape.op != "tensors.reshape":
        return None
    source = module.node_map[reshape.inputs[0]]
    if (
        not isinstance(source.type, TensorType)
        or not isinstance(reshape.type, TensorType)
        or not isinstance(node.type, TensorType)
    ):
        return None
    input_shape = _fixed_shape(source.type)
    output_shape = _fixed_shape(reshape.type)
    packed_shape = _fixed_shape(node.type)
    if input_shape is None or output_shape is None or packed_shape is None:
        return None
    output_axes, lanes = _pack_contract(node, reshape.type.rank)
    input_axes = _map_lane_axes(output_shape, output_axes, lanes, input_shape)
    if input_axes is None:
        return None
    return reshape, source, input_axes, lanes, packed_shape


def _pack_reshape_matches(node: Node, module: IRModule) -> bool:
    return _pack_reshape_plan(node, module) is not None


def _pack_reshape(node: Node, module: IRModule) -> RewriteResult:
    plan = _pack_reshape_plan(node, module)
    assert plan is not None
    reshape, source, input_axes, lanes, packed_shape = plan
    packed = _make(
        f"{node.id}.propagated.pack",
        "tensors.pack",
        (source,),
        {"axes": input_axes, "lanes": lanes},
        propagation_helper_metadata(node, node.id, "propagated-pack"),
    )
    replacement = _make(
        node.id,
        "tensors.reshape",
        (packed,),
        {"shape": packed_shape},
        propagation_result_metadata(
            reshape,
            node,
            axes=input_axes,
            lanes=lanes,
            rule="VectorizeReshapePropagation",
            internal_role="propagated-reshape",
        ),
    )
    return RewriteResult(replacement, (packed,))


def _reshape_unpack_plan(node: Node, module: IRModule):
    if node.op != "tensors.reshape" or not isinstance(node.type, TensorType):
        return None
    unpack = module.node_map[node.inputs[0]]
    if unpack.op != "tensors.unpack" or not isinstance(unpack.type, TensorType):
        return None
    contract = _unpack_contract(unpack, module)
    if contract is None:
        return None
    vector, input_axes, lanes = contract
    input_shape = _fixed_shape(unpack.type)
    output_shape = _fixed_shape(node.type)
    if input_shape is None or output_shape is None:
        return None
    output_axes = _map_lane_axes(input_shape, input_axes, lanes, output_shape)
    if output_axes is None:
        return None
    vector_output_shape = list(output_shape)
    for axis, lane in zip(output_axes, lanes):
        if vector_output_shape[axis] % lane:
            return None
        vector_output_shape[axis] //= lane
    return unpack, vector, output_axes, lanes, tuple(vector_output_shape)


def _reshape_unpack_matches(node: Node, module: IRModule) -> bool:
    return _reshape_unpack_plan(node, module) is not None


def _reshape_unpack(node: Node, module: IRModule) -> RewriteResult:
    plan = _reshape_unpack_plan(node, module)
    assert plan is not None
    unpack, vector, output_axes, lanes, vector_output_shape = plan
    reshaped = _make(
        f"{node.id}.propagated.reshape",
        "tensors.reshape",
        (vector,),
        {"shape": vector_output_shape},
        propagation_result_metadata(
            node,
            unpack,
            axes=output_axes,
            lanes=lanes,
            rule="ReshapeDevectorizePropagation",
            internal_role="propagated-reshape",
        ),
    )
    replacement = _make(
        node.id,
        "tensors.unpack",
        (reshaped,),
        {"axes": output_axes},
        propagation_result_metadata(
            node,
            unpack,
            axes=output_axes,
            lanes=lanes,
            rule="ReshapeDevectorizePropagation",
        ),
    )
    return RewriteResult(replacement, (reshaped,))


def reshape_propagation_rules() -> tuple[RewriteRule, ...]:
    return (
        RewriteRule("VectorizeReshapePropagation", _pack_reshape_matches, _pack_reshape),
        RewriteRule("ReshapeDevectorizePropagation", _reshape_unpack_matches, _reshape_unpack),
    )


__all__ = ["reshape_propagation_rules"]
