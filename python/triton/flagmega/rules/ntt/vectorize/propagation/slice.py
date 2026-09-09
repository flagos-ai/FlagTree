# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nncase aligned Slice/Pack propagation, in both boundary directions."""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.tensors.pack import axis_lane_products, normalize_axes
from triton.flagmega.ir.ops.tensors.slice import Slice
from triton.flagmega.ir.types import VectorType
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.neutral._utility import make_node
from triton.flagmega.rules.ntt.vectorize.utility import propagation_helper_metadata, propagation_result_metadata


def _scaled_ranges(source, attrs, axes, lanes):
    """Normalize Python slice bounds before dividing the packed coordinates.

    Steps on packed axes must be one: a strided/reversed slice changes lane
    order, not just vector coordinates. Unrelated axes keep their semantics.
    """
    tensor = tensor_of(source.type)
    ranges = dict(Slice.ranges(tensor, attrs))
    factors = axis_lane_products(lanes, axes)
    starts, ends = list(attrs["starts"]), list(attrs["ends"])
    for i, axis in enumerate(normalize_axes(tuple(attrs["axes"]), tensor.rank)):
        if axis not in factors or axis not in ranges:
            continue
        indices, factor = ranges[axis], factors[axis]
        if indices.step != 1 or indices.start % factor or indices.stop % factor:
            return None
        starts[i], ends[i] = indices.start // factor, indices.stop // factor
    return {**dict(attrs), "starts": tuple(starts), "ends": tuple(ends)}


def _pack_slice(node, module):
    sliced = module.node_map[node.inputs[0]]
    if sliced.op != Slice.op_name or not node.effect.is_pure or not sliced.effect.is_pure:
        return None
    source = module.node_map[sliced.inputs[0]]
    lanes = tuple(node.attrs["lanes"])
    axes = normalize_axes(tuple(node.attrs.get("axes", (node.attrs.get("axis", -1), ) * len(lanes))),
                          tensor_of(source.type).rank)
    attrs = _scaled_ranges(source, sliced.attrs, axes, lanes)
    if attrs is None:
        return None
    try:
        packed = make_node("tensors.pack", f"{node.id}.propagated.pack", (source, ), {"axes": axes, "lanes": lanes},
                           propagation_helper_metadata(node, node.id, "propagated-pack"))
        result = make_node(
            Slice.op_name, node.id, (packed, ), attrs,
            propagation_result_metadata(sliced, node, axes=axes, lanes=lanes, rule="VectorizeSlicePropagation",
                                        internal_role="propagated-slice"))
    except IRSchemaError:
        # The full input, unlike the slice result, may not be packable, or
        # its distributed split may cut lanes. Do not insert padding/boxing.
        return None
    return RewriteResult(result, (packed, )) if result.type == node.type else None


def _slice_unpack(node, module):
    unpack = module.node_map[node.inputs[0]]
    if unpack.op != "tensors.unpack" or not node.effect.is_pure or not unpack.effect.is_pure:
        return None
    vector = module.node_map[unpack.inputs[0]]
    dtype = tensor_of(vector.type).dtype
    if not isinstance(dtype, VectorType):
        return None
    raw_axes = tuple(unpack.attrs.get("axes", (unpack.attrs.get("axis", -1), ) * len(dtype.lanes)))
    axes = normalize_axes(raw_axes, tensor_of(vector.type).rank)
    lanes = dtype.lanes[:len(axes)]
    attrs = _scaled_ranges(unpack, node.attrs, axes, lanes)
    if attrs is None:
        return None
    try:
        sliced = make_node(
            Slice.op_name, f"{node.id}.propagated.slice", (vector, ), attrs,
            propagation_result_metadata(node, unpack, axes=axes, lanes=lanes, rule="SliceDevectorizePropagation",
                                        internal_role="propagated-slice"))
        result = make_node(
            "tensors.unpack", node.id, (sliced, ), {"axes": axes},
            propagation_result_metadata(node, unpack, axes=axes, lanes=lanes, rule="SliceDevectorizePropagation"))
    except IRSchemaError:
        return None
    return RewriteResult(result, (sliced, )) if result.type == node.type else None


def slice_propagation_rules():
    return (
        RewriteRule("VectorizeSlicePropagation", lambda node, _: node.op == "tensors.pack", _pack_slice),
        RewriteRule("SliceDevectorizePropagation", lambda node, _: node.op == Slice.op_name, _slice_unpack),
    )


__all__ = ["slice_propagation_rules"]
