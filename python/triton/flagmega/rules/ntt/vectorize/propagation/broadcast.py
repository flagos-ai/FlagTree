# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Propagate Pack into broadcast coordinates and vector element lanes."""

from dataclasses import replace

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.tensors.pack import normalize_axes
from triton.flagmega.ir.types import VectorType
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.neutral._utility import make_node
from triton.flagmega.rules.ntt.vectorize.utility import propagation_helper_metadata, propagation_result_metadata


def _pack_broadcast(node, module):
    broadcast = module.node_map[node.inputs[0]]
    if broadcast.op != "tensors.broadcast_to" or not node.effect.is_pure or not broadcast.effect.is_pure:
        return None
    source = module.node_map[broadcast.inputs[0]]
    input_tensor, expanded, output = tensor_of(source.type), tensor_of(broadcast.type), tensor_of(node.type)
    lanes = tuple(node.attrs["lanes"])
    axes = normalize_axes(tuple(node.attrs.get("axes", (node.attrs.get("axis", -1), ) * len(lanes))), expanded.rank)
    offset = expanded.rank - input_tensor.rank
    source_axes, source_lanes, aligned_lanes = [], [], []
    for axis, lane in zip(axes, lanes):
        input_axis = axis - offset
        if input_axis < 0 or (input_tensor.shape[input_axis].is_fixed
                              and input_tensor.shape[input_axis].fixed_value == 1):
            aligned_lanes.append(1)
        else:
            source_axes.append(input_axis)
            source_lanes.append(lane)
            aligned_lanes.append(lane)
    helpers = []
    metadata = propagation_helper_metadata(node, node.id, "propagated-pack")
    try:
        operand = source
        if source_lanes:
            operand = make_node("tensors.pack", f"{node.id}.propagated.pack", (source, ),
                                {"axes": tuple(source_axes), "lanes": tuple(source_lanes)}, metadata)
            helpers.append(operand)
        # Insert unit *element* dimensions for new broadcast lanes, separately
        # from tensor dimensions. This view changes neither bytes nor tensor
        # shape; it is not a replacement for any activation Pack.
        old_lanes = getattr(input_tensor.dtype, "lanes", ())
        expanded_old_lanes = getattr(expanded.dtype, "lanes", ())
        aligned = (*aligned_lanes, *((1, ) * (len(expanded_old_lanes) - len(old_lanes))), *old_lanes)
        operand_tensor = tensor_of(operand.type)
        # Ordinary trailing broadcasting already inserts leading unit lanes.
        desired_lanes = aligned
        while desired_lanes and desired_lanes[0] == 1:
            desired_lanes = desired_lanes[1:]
        element_type = input_tensor.dtype.elem_type if isinstance(input_tensor.dtype,
                                                                  VectorType) else input_tensor.dtype
        desired_dtype = VectorType(element_type, desired_lanes) if desired_lanes else element_type
        if operand_tensor.dtype != desired_dtype:
            view = make_node("tensors.bitcast", f"{node.id}.propagated.lane_dims", (operand, ),
                             {"dtype": desired_dtype}, metadata)
            view_tensor = tensor_of(view.type)
            if view_tensor.shape != operand_tensor.shape or view_tensor.layout != operand_tensor.layout:
                return None
            helpers.append(view)
            operand = view
        result = make_node(
            "tensors.broadcast_to", node.id, (operand, ),
            {"shape": tuple(d.fixed_value for d in output.shape), "output_lanes": output.dtype.lanes},
            propagation_result_metadata(broadcast, node, axes=axes, lanes=lanes, rule="VectorizeBroadcastPropagation",
                                        internal_role="propagated-broadcast"))
    except IRSchemaError:
        return None
    return RewriteResult(result, tuple(helpers)) if result.type == node.type else None


def _broadcast_unpack(node, module):
    unpack = module.node_map[node.inputs[0]]
    if unpack.op != "tensors.unpack" or not node.effect.is_pure or not unpack.effect.is_pure:
        return None
    vector = module.node_map[unpack.inputs[0]]
    tensor = tensor_of(vector.type)
    axes = normalize_axes(tuple(unpack.attrs.get("axes", (unpack.attrs.get("axis", -1), ) * len(tensor.dtype.lanes))),
                          tensor.rank)
    result_axes = tuple(axis + tensor_of(node.type).rank - tensor.rank for axis in axes)
    lanes = tensor.dtype.lanes[:len(axes)]
    try:
        # Reuse the forward proof (including lane rank alignment and SBP),
        # then cancel its projected Pack against the existing input Unpack.
        demand = make_node("tensors.pack", f"{node.id}.propagated.broadcast", (node, ),
                           {"axes": result_axes, "lanes": lanes}, {})
        packed = _pack_broadcast(demand, module)
        if packed is None:
            return None
        aliases = {}
        helpers = []
        for helper in packed.prefix_nodes:
            if (helper.op == "tensors.pack" and helper.inputs == (unpack.id, ) and helper.type == vector.type
                    and tuple(helper.attrs["axes"]) == axes and tuple(helper.attrs["lanes"]) == lanes):
                aliases[helper.id] = vector.id
            else:
                helpers.append(replace(helper, inputs=tuple(aliases.get(name, name) for name in helper.inputs)))
        broadcast = replace(packed.replacement,
                            inputs=tuple(aliases.get(name, name) for name in packed.replacement.inputs))
        result = make_node(
            "tensors.unpack", node.id, (broadcast, ), {"axes": result_axes},
            propagation_result_metadata(node, unpack, axes=result_axes, lanes=lanes,
                                        rule="BroadcastDevectorizePropagation"))
    except IRSchemaError:
        return None
    return RewriteResult(result, (*helpers, broadcast)) if result.type == node.type else None


def broadcast_propagation_rules():
    return (
        RewriteRule("VectorizeBroadcastPropagation", lambda node, _: node.op == "tensors.pack", _pack_broadcast),
        RewriteRule("BroadcastDevectorizePropagation", lambda node, _: node.op == "tensors.broadcast_to",
                    _broadcast_unpack),
    )


__all__ = ["broadcast_propagation_rules"]
