# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nncase Gather packing: feature Pack belongs on the embedding table.

Rebuild at the producer so shared scalar consumers do not duplicate the
lookup. Restoring the old result type keeps function outputs and other users
valid; ordinary equalities remove Pack(Unpack(...)) at packed consumers.
"""

from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.neutral._utility import make_node
from triton.flagmega.rules.ntt.vectorize.propagation.producer_demand import final_axis_pack_demand


def embedding_producer_rule():

    def rewrite(node, module):
        lanes = final_axis_pack_demand(module, {node.id})
        if lanes is None:
            return None
        indices, weight = (module.node_map[name] for name in node.inputs)
        metadata = {"rewritten_by": "VectorizeEmbeddingPropagation"}
        packed_weight = make_node("tensors.pack", f"{node.id}.packed_weight", (weight, ),
                                  {"axes": (1, ) * len(lanes), "lanes": lanes}, metadata)
        packed = make_node(node.op, f"{node.id}.packed", (indices, packed_weight), node.attrs,
                           {**dict(node.metadata), **metadata})
        rank = tensor_of(node.type).rank
        restored = make_node("tensors.unpack", node.id, (packed, ), {"axes": (rank - 1, ) * len(lanes)}, metadata)
        assert restored.type == node.type
        return RewriteResult(restored, (packed_weight, packed))

    return RewriteRule("VectorizeEmbeddingPropagation", lambda node, _: node.op == "nn.embedding", rewrite)


__all__ = ["embedding_producer_rule"]
