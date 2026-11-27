# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Pack KDA's activation producer once without changing its state write order."""

from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.rules import RewriteEffectPolicy, RewriteResult, RewriteRule
from triton.flagmega.rules.neutral._utility import make_node
from triton.flagmega.rules.ntt.vectorize.propagation.producer_demand import final_axis_pack_demand


def kda_recurrent_producer_rule():
    def rewrite(node, module):
        projections = {
            value.id
            for value in module.nodes
            if value.op == "builtin.get_item" and value.inputs == (node.id,) and value.attrs["index"] == 0
        }
        lanes = final_axis_pack_demand(module, projections)
        if lanes is None:
            return None
        metadata = {"rewritten_by": "VectorizeKdaRecurrentPropagation"}
        packed = make_node(
            node.op,
            f"{node.id}.packed",
            tuple(module.node_map[name] for name in node.inputs),
            {**dict(node.attrs), "output_lanes": (*lanes, *node.attrs.get("output_lanes", ()))},
            {**dict(node.metadata), **metadata},
        )
        helpers, restored = [packed], []
        for index in range(2):
            field = make_node("builtin.get_item", f"{packed.id}.{index}", (packed,), {"index": index}, metadata)
            helpers.append(field)
            if index == 0:
                field = make_node(
                    "tensors.unpack",
                    f"{field.id}.unpack",
                    (field,),
                    {"axes": (tensor_of(node.type.fields[0]).rank - 1,) * len(lanes)},
                    metadata,
                )
                helpers.append(field)
            restored.append(field)
        result = make_node("builtin.tuple", node.id, tuple(restored), {}, {**dict(node.metadata), **metadata})
        assert result.type == node.type and packed.effect == node.effect
        return RewriteResult(result, tuple(helpers))

    return RewriteRule(
        "VectorizeKdaRecurrentPropagation",
        lambda node, _: node.op in {"nn.kda_recurrent_core", "nn.kda_gated_recurrent_core", "nn.kda_gated_step"},
        rewrite,
        effect_policy=RewriteEffectPolicy.ALLOW,
        root_ops=("nn.kda_recurrent_core", "nn.kda_gated_recurrent_core", "nn.kda_gated_step"),
    )


__all__ = ["kda_recurrent_producer_rule"]
