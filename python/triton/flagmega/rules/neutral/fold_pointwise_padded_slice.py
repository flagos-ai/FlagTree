# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Discarded tail lanes need not be reinitialized before pointwise compute.

crop(f(pack(pad(crop(unpack(v)))))) == crop(f(v)) when both crops
cover the same prefix and the padded representation restores v's type.
Only the cropped root is an equality: the whole vectors need not be equal.
"""

from triton.flagmega.ir import TensorType, VectorType
from triton.flagmega.ir.ops.tensors.pack import normalize_axes
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.neutral._utility import make_node


def fold_pointwise_padded_slice_rule():

    def rewrite(node, module):
        nodes = module.node_map
        view = nodes[node.inputs[0]]
        if view.op != "tensors.bitcast":
            return None
        unary = nodes[view.inputs[0]]
        if unary.op != "math.vectorized_unary" or not unary.effect.is_pure:
            return None
        pack = nodes[unary.inputs[0]]
        if pack.op != "tensors.pack":
            return None
        pad = nodes[pack.inputs[0]]
        if pad.op != "tensors.pad":
            return None
        crop = nodes[pad.inputs[0]]
        if crop.op != "tensors.slice_to_shape" or crop.attrs["shape"] != node.attrs["shape"]:
            return None
        original_view = nodes[crop.inputs[0]]
        if original_view.op != "tensors.bitcast":
            return None
        if any(not value.effect.is_pure for value in (node, view, pack, pad, crop, original_view)):
            return None
        vector = nodes[original_view.inputs[0]]
        if (not isinstance(vector.type, TensorType) or not isinstance(vector.type.dtype, VectorType)
                or vector.type != pack.type or original_view.type != view.type
                or vector.type.dtype.elem_type != original_view.type.dtype):
            return None
        lanes = tuple(pack.attrs["lanes"])
        axes = normalize_axes(tuple(pack.attrs.get("axes", (pack.attrs.get("axis", -1), ) * len(lanes))),
                              vector.type.rank)
        if lanes != vector.type.dtype.lanes or any(axis != vector.type.rank - 1 for axis in axes):
            return None
        compute = make_node(unary.op, f"{node.id}.uncropped.compute", (vector, ), unary.attrs, unary.metadata)
        restored = make_node(view.op, f"{node.id}.uncropped.view", (compute, ), view.attrs, view.metadata)
        result = make_node(node.op, node.id, (restored, ), node.attrs,
                           {**dict(node.metadata), "rewritten_by": "FoldPointwisePaddedSlice"})
        return RewriteResult(result, (compute, restored))

    return RewriteRule("FoldPointwisePaddedSlice", lambda node, _: node.op == "tensors.slice_to_shape", rewrite)


__all__ = ["fold_pointwise_padded_slice_rule"]
