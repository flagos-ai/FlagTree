# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Pack the cos/sin producer without cloning or moving a state read."""

from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.rules import RewriteEffectPolicy, RewriteResult, RewriteRule
from triton.flagmega.rules.neutral._utility import make_node
from triton.flagmega.rules.ntt.vectorize.propagation.producer_demand import final_axis_pack_demand


def rotary_embedding_producer_rule():

    def rewrite(node, module):
        inputs = tuple(module.node_map[name] for name in node.inputs)
        reference = inputs[0]
        # reference has MemoryEffect.NONE: only its sequence extent is used.
        # Feature layout views do not change that extent, even for vectors.
        while reference.op in {"tensors.pack", "tensors.unpack", "tensors.bitcast"}:
            source = module.node_map[reference.inputs[0]]
            old, new = tensor_of(reference.type), tensor_of(source.type)
            if old.rank != 2 or new.rank != 2 or old.shape[0] != new.shape[0]:
                break
            reference = source
        inputs = (reference, inputs[1])
        projections = {
            value.id
            for value in module.nodes
            if value.op == "builtin.get_item" and value.inputs == (node.id, )
        }
        lanes = final_axis_pack_demand(module, projections)
        if lanes is None:
            return make_node(node.op, node.id, inputs, node.attrs,
                             node.metadata) if reference.id != node.inputs[0] else None
        metadata = {"rewritten_by": "VectorizeRotaryEmbeddingPropagation"}
        packed = make_node(node.op, f"{node.id}.packed", inputs,
                           {**dict(node.attrs), "output_lanes":
                            (*lanes, *node.attrs.get("output_lanes", ()))}, {**dict(node.metadata), **metadata})
        helpers = [packed]
        restored = []
        for index in range(2):
            field = make_node("builtin.get_item", f"{packed.id}.{index}", (packed, ), {"index": index}, metadata)
            unpack = make_node("tensors.unpack", f"{field.id}.unpack", (field, ), {"axes": (2, ) * len(lanes)},
                               metadata)
            helpers.extend((field, unpack))
            restored.append(unpack)
        result = make_node("builtin.tuple", node.id, tuple(restored), {}, metadata)
        assert result.type == node.type and packed.effect == node.effect
        # Replace the original effectful root, not a later Pack user: exactly
        # one read stays at this point in the effect order, before any writes.
        return RewriteResult(result, tuple(helpers))

    return RewriteRule("VectorizeRotaryEmbeddingPropagation", lambda node, _: node.op == "nn.rotary_embedding", rewrite,
                       effect_policy=RewriteEffectPolicy.ALLOW)


__all__ = ["rotary_embedding_producer_rule"]
