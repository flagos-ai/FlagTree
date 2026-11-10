# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Preserve one two-result projection while satisfying per-field lane demands."""

from triton.flagmega.ir import VectorType
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.neutral._utility import make_node
from triton.flagmega.rules.ntt.vectorize.propagation.producer_demand import final_axis_pack_demand


def interleaved_projection_producer_rule():
    def rewrite(node, module):
        demands = tuple(final_axis_pack_demand(module, {
            value.id for value in module.nodes if value.op == "builtin.get_item"
            and value.inputs == (node.id,) and value.attrs["index"] == index
        }) for index in range(2))
        if not any(demands):
            return None
        previous = node.attrs.get("output_lanes", ((),) * 2)
        lanes = tuple((*demand, *old) if demand else old
                      for demand, old in zip(demands, previous, strict=True))
        metadata = {"rewritten_by": "VectorizeInterleavedProjectionPropagation"}
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

    return RewriteRule("VectorizeInterleavedProjectionPropagation",
                       lambda node, _: node.op == "nn.interleaved_projection", rewrite,
                       root_ops=("nn.interleaved_projection",))



__all__ = ["interleaved_projection_producer_rule"]
