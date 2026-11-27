# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Make one MhcPre emit the collapsed layout demanded by its consumers."""

from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.neutral._utility import make_node
from triton.flagmega.rules.ntt.vectorize.propagation.producer_demand import final_axis_pack_demand


def mhc_pre_producer_rule():
    def rewrite(node, module):
        projections = {
            value.id
            for value in module.nodes
            if value.op == "builtin.get_item" and value.inputs == (node.id,) and value.attrs["index"] == 2
        }
        lanes = final_axis_pack_demand(module, projections)
        if lanes is None:
            return None
        metadata = {"rewritten_by": "VectorizeMhcPrePropagation"}
        inputs = tuple(module.node_map[name] for name in node.inputs)
        packed = make_node(
            node.op,
            f"{node.id}.packed",
            inputs,
            {**dict(node.attrs), "output_lanes": (*lanes, *node.attrs.get("output_lanes", ()))},
            {**dict(node.metadata), **metadata},
        )
        helpers, restored = [packed], []
        for index in range(3):
            field = make_node("builtin.get_item", f"{packed.id}.{index}", (packed,), {"index": index}, metadata)
            helpers.append(field)
            if index == 2:
                rank = tensor_of(node.type.fields[index]).rank
                field = make_node(
                    "tensors.unpack", f"{field.id}.unpack", (field,), {"axes": (rank - 1,) * len(lanes)}, metadata
                )
                helpers.append(field)
            restored.append(field)
        # Preserve the tuple ABI for scalar/mixed-layout and whole-tuple users.
        # Matching consumers fold Pack(Unpack(...)) to the packed projection.
        result = make_node("builtin.tuple", node.id, tuple(restored), {}, {**dict(node.metadata), **metadata})
        assert result.type == node.type and packed.effect == node.effect
        return RewriteResult(result, tuple(helpers))

    return RewriteRule(
        "VectorizeMhcPrePropagation", lambda node, _: node.op == "nn.mhc_pre", rewrite, root_ops=("nn.mhc_pre",)
    )


__all__ = ["mhc_pre_producer_rule"]
