# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Preserve one six-result projection while satisfying per-field lane demands."""

from triton.flagmega.ir import VectorType
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.neutral._utility import make_node
from triton.flagmega.rules.ntt.vectorize.propagation.producer_demand import final_axis_pack_demand


def kda_input_projection_producer_rule():
    def rewrite(node, module):
        demands = tuple(final_axis_pack_demand(module, {
            value.id for value in module.nodes if value.op == "builtin.get_item"
            and value.inputs == (node.id,) and value.attrs["index"] == index
        }) for index in range(6))
        if not any(demands):
            return None
        previous = node.attrs.get("output_lanes", ((),) * 6)
        lanes = tuple((*demand, *old) if demand else old
                      for demand, old in zip(demands, previous, strict=True))
        metadata = {"rewritten_by": "VectorizeKdaInputProjectionPropagation"}
        packed = make_node(node.op, f"{node.id}.packed", tuple(module.node_map[name] for name in node.inputs),
                           {**dict(node.attrs), "output_lanes": lanes}, {**dict(node.metadata), **metadata})
        helpers, restored = [packed], []
        for index, demand in enumerate(demands):
            field = make_node("builtin.get_item", f"{packed.id}.{index}", (packed,), {"index": index}, metadata)
            helpers.append(field)
            if demand:
                field = make_node("tensors.unpack", f"{field.id}.unpack", (field,),
                                  {"axes": (tensor_of(node.type.fields[index]).rank - 1,) * len(demand)}, metadata)
                helpers.append(field)
            restored.append(field)
        result = make_node("builtin.tuple", node.id, tuple(restored), {}, {**dict(node.metadata), **metadata})
        assert result.type == node.type and packed.effect == node.effect
        return RewriteResult(result, tuple(helpers))

    return RewriteRule("VectorizeKdaInputProjectionPropagation",
                       lambda node, _: node.op == "nn.kda_input_projection", rewrite,
                       root_ops=("nn.kda_input_projection",))


def kda_input_projection_input_rule():
    def rewrite(node, module):
        unpack = module.node_map[node.inputs[0]]
        if unpack.op not in {"tensors.unpack", "tensors.bitcast"}:
            return None
        source = module.node_map[unpack.inputs[0]]
        source_type, result_type = tensor_of(source.type), tensor_of(unpack.type)
        if (not isinstance(source_type.dtype, VectorType)
                or source_type.dtype.elem_type != result_type.dtype
                or source_type.shape[:-1] != result_type.shape[:-1]
                or source_type.shape[-1].fixed_value * source_type.dtype.lane_count != result_type.shape[-1].fixed_value):
            return None
        rank = tensor_of(unpack.type).rank
        axes = tuple(unpack.attrs.get("axes", (unpack.attrs.get("axis", -1),)))
        if unpack.op == "tensors.unpack" and (not axes or any(axis % rank != rank - 1 for axis in axes)):
            return None
        inputs = (source, *(module.node_map[name] for name in node.inputs[1:]))
        result = make_node(node.op, node.id, inputs, dict(node.attrs), dict(node.metadata))
        assert result.type == node.type
        return RewriteResult(result)

    return RewriteRule("VectorizeFinalAxisProjectionInput",
                       lambda node, _: node.op in {"nn.kda_input_projection", "nn.partitioned_linear",
                                                  "nn.partitioned_block_scaled_linear"}, rewrite,
                       root_ops=("nn.kda_input_projection", "nn.partitioned_linear", "nn.partitioned_block_scaled_linear"))
