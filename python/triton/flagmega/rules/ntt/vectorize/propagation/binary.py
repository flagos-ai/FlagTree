# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Push Pack/Unpack through binary dataflow."""

from __future__ import annotations

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import IRModule, Node, TensorType, VectorType
from triton.flagmega.ir.ops.tensors.pack import Pack
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.ntt.vectorize.utility import (
    propagation_helper_metadata,
    propagation_result_metadata,
)


def _pack_binary_matches(node: Node, module: IRModule) -> bool:
    return node.op == "tensors.pack" and module.node_map[node.inputs[0]].op in {"math.add", "math.mul"}


def _pack_binary(node: Node, module: IRModule) -> RewriteResult:
    scalar = module.node_map[node.inputs[0]]
    boundary_lanes = tuple(int(value) for value in node.attrs["lanes"])
    boundary_axes = _pack_axes(node, len(boundary_lanes))
    helpers: list[Node] = []
    packed: list[Node] = []
    for index, input_id in enumerate(scalar.inputs):
        source = module.node_map[input_id]
        attrs = dict(node.attrs)
        value_type = Pack.infer_type((source,), attrs)
        helper = Node(
            f"{node.id}.propagated.pack{index}",
            "tensors.pack",
            (source.id,),
            value_type,
            attrs=attrs,
            metadata=propagation_helper_metadata(node, node.id, "propagated-pack"),
        )
        helpers.append(helper)
        packed.append(helper)
    replacement = Node(
        node.id,
        "math.vectorized_binary",
        tuple(value.id for value in packed),
        node.type,
        attrs={"binary_op": scalar.op.removeprefix("math.")},
        metadata=propagation_result_metadata(
            scalar,
            node,
            axes=boundary_axes,
            lanes=boundary_lanes,
            rule="VectorizeBinaryPropagation",
            internal_role="propagated-compute",
        ),
    )
    return RewriteResult(replacement, tuple(helpers))


def _binary_unpack_matches(node: Node, module: IRModule, operand_index: int) -> bool:
    if node.op not in {"math.add", "math.mul"} or len(node.inputs) != 2:
        return False
    unpack = module.node_map[node.inputs[operand_index]]
    if unpack.op != "tensors.unpack":
        return False
    vector = module.node_map[unpack.inputs[0]]
    other = module.node_map[node.inputs[1 - operand_index]]
    if (
        not isinstance(vector.type, TensorType) or not isinstance(vector.type.dtype, VectorType)
        or not isinstance(other.type, TensorType) or isinstance(other.type.dtype, VectorType)
    ):
        return False
    axes = _unpack_axes(unpack, len(vector.type.dtype.lanes))
    lanes = vector.type.dtype.lanes[:len(axes)]
    try:
        packed_type = Pack.infer_type((other,), {"lanes": lanes, "axes": axes})
    except (IRSchemaError, TypeError, ValueError):
        return False
    return packed_type == vector.type


def _binary_unpack(node: Node, module: IRModule, operand_index: int) -> RewriteResult:
    unpack = module.node_map[node.inputs[operand_index]]
    vector = module.node_map[unpack.inputs[0]]
    other = module.node_map[node.inputs[1 - operand_index]]
    assert isinstance(vector.type, TensorType) and isinstance(vector.type.dtype, VectorType)
    axes = _unpack_axes(unpack, len(vector.type.dtype.lanes))
    lanes = vector.type.dtype.lanes[:len(axes)]
    pack_attrs = {"lanes": lanes, "axes": axes}
    packed_other = Node(
        f"{node.id}.propagated.pack{1 - operand_index}",
        "tensors.pack",
        (other.id,),
        Pack.infer_type((other,), pack_attrs),
        attrs=pack_attrs,
        metadata=propagation_helper_metadata(unpack, node.id, "propagated-pack"),
    )
    vector_inputs = [packed_other.id, packed_other.id]
    vector_inputs[operand_index] = vector.id
    compute = Node(
        f"{node.id}.propagated.compute",
        "math.vectorized_binary",
        tuple(vector_inputs),
        vector.type,
        attrs={"binary_op": node.op.removeprefix("math.")},
        metadata=propagation_result_metadata(
            node,
            unpack,
            axes=axes,
            lanes=lanes,
            rule="BinaryDevectorizePropagation",
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
            lanes=lanes,
            rule="BinaryDevectorizePropagation",
        ),
    )
    return RewriteResult(replacement, (packed_other, compute))


def _unpack_axes(unpack: Node, lane_count: int) -> tuple[int, ...]:
    if "axes" in unpack.attrs:
        return tuple(int(value) for value in unpack.attrs["axes"])
    return (int(unpack.attrs["axis"]),) * lane_count


def _pack_axes(pack: Node, lane_count: int) -> tuple[int, ...]:
    if "axes" in pack.attrs:
        return tuple(int(value) for value in pack.attrs["axes"])
    return (int(pack.attrs["axis"]),) * lane_count


def binary_propagation_rules() -> tuple[RewriteRule, ...]:
    return (
        RewriteRule("VectorizeBinaryPropagation", _pack_binary_matches, _pack_binary),
        RewriteRule(
            "BinaryDevectorizeLhsPropagation",
            lambda node, module: _binary_unpack_matches(node, module, 0),
            lambda node, module: _binary_unpack(node, module, 0),
        ),
        RewriteRule(
            "BinaryDevectorizeRhsPropagation",
            lambda node, module: _binary_unpack_matches(node, module, 1),
            lambda node, module: _binary_unpack(node, module, 1),
        ),
    )


__all__ = ["binary_propagation_rules"]
