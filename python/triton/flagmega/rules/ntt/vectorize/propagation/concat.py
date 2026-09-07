# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Propagate typed vector boundaries through variadic concat."""

from __future__ import annotations

from triton.flagmega.ir import IRModule, Node, TensorType, VectorType, get_definition, try_div_exactly
from triton.flagmega.ir.ops.tensors.pack import normalize_axes
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.ntt.vectorize.utility import (
    propagation_helper_metadata,
    propagation_result_metadata,
)


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


def _packable(value: Node, axes: tuple[int, ...], lanes: tuple[int, ...]) -> bool:
    return isinstance(value.type, TensorType) and all(
        try_div_exactly(value.type.shape[axis], lane) is not None
        for axis, lane in zip(axes, lanes)
    )


def _pack_concat_matches(node: Node, module: IRModule) -> bool:
    if node.op != "tensors.pack":
        return False
    concat = module.node_map[node.inputs[0]]
    if concat.op != "tensors.concat" or not isinstance(concat.type, TensorType):
        return False
    axes, lanes = _pack_contract(node, concat.type.rank)
    return all(_packable(module.node_map[value], axes, lanes) for value in concat.inputs)


def _pack_concat(node: Node, module: IRModule) -> RewriteResult:
    concat = module.node_map[node.inputs[0]]
    assert isinstance(concat.type, TensorType)
    axes, lanes = _pack_contract(node, concat.type.rank)
    helpers = tuple(
        _make(
            f"{node.id}.propagated.pack{index}",
            "tensors.pack",
            (module.node_map[input_id],),
            {"axes": axes, "lanes": lanes},
            propagation_helper_metadata(node, node.id, "propagated-pack"),
        )
        for index, input_id in enumerate(concat.inputs)
    )
    replacement = _make(
        node.id,
        "tensors.concat",
        helpers,
        concat.attrs,
        propagation_result_metadata(
            concat,
            node,
            axes=axes,
            lanes=lanes,
            rule="VectorizeConcatPropagation",
            internal_role="propagated-concat",
        ),
    )
    return RewriteResult(replacement, helpers)


def _concat_unpack_contract(node: Node, module: IRModule):
    if node.op != "tensors.concat":
        return None
    contracts = [
        _unpack_contract(value, module)
        for input_id in node.inputs
        if (value := module.node_map[input_id]).op == "tensors.unpack"
    ]
    if not contracts or any(value is None for value in contracts):
        return None
    first = contracts[0]
    assert first is not None
    axes, lanes = first[1], first[2]
    if any(value[1:] != (axes, lanes) for value in contracts if value is not None):
        return None
    for input_id in node.inputs:
        value = module.node_map[input_id]
        if value.op != "tensors.unpack" and not _packable(value, axes, lanes):
            return None
    return axes, lanes


def _concat_unpack_matches(node: Node, module: IRModule) -> bool:
    return _concat_unpack_contract(node, module) is not None


def _concat_unpack(node: Node, module: IRModule) -> RewriteResult:
    contract = _concat_unpack_contract(node, module)
    assert contract is not None
    axes, lanes = contract
    boundary = next(
        module.node_map[candidate]
        for candidate in node.inputs
        if module.node_map[candidate].op == "tensors.unpack"
    )
    helpers: list[Node] = []
    vector_inputs: list[Node] = []
    for index, input_id in enumerate(node.inputs):
        value = module.node_map[input_id]
        if value.op == "tensors.unpack":
            vector_inputs.append(module.node_map[value.inputs[0]])
            continue
        packed = _make(
            f"{node.id}.propagated.pack{index}",
            "tensors.pack",
            (value,),
            {"axes": axes, "lanes": lanes},
            propagation_helper_metadata(boundary, node.id, "propagated-pack"),
        )
        helpers.append(packed)
        vector_inputs.append(packed)
    vector_concat = _make(
        f"{node.id}.propagated.concat",
        "tensors.concat",
        tuple(vector_inputs),
        node.attrs,
        propagation_result_metadata(
            node,
            boundary,
            axes=axes,
            lanes=lanes,
            rule="ConcatDevectorizePropagation",
            internal_role="propagated-concat",
        ),
    )
    helpers.append(vector_concat)
    replacement = _make(
        node.id,
        "tensors.unpack",
        (vector_concat,),
        {"axes": axes},
        propagation_result_metadata(
            node,
            boundary,
            axes=axes,
            lanes=lanes,
            rule="ConcatDevectorizePropagation",
        ),
    )
    return RewriteResult(replacement, tuple(helpers))


def concat_propagation_rules() -> tuple[RewriteRule, ...]:
    return (
        RewriteRule("VectorizeConcatPropagation", _pack_concat_matches, _pack_concat),
        RewriteRule("ConcatDevectorizePropagation", _concat_unpack_matches, _concat_unpack),
    )


__all__ = ["concat_propagation_rules"]
