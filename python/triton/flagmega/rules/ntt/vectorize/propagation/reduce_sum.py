# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Push final-axis layout demands into the reduction's existing output store."""

from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.neutral._utility import make_node
from triton.flagmega.rules.ntt.vectorize.propagation.producer_demand import final_axis_pack_demand


def reduce_sum_producer_rule():
    def rewrite(node, module):
        lanes = final_axis_pack_demand(module, {node.id})
        if lanes is None:
            return None
        metadata = {**dict(node.metadata), "rewritten_by": "VectorizeReduceSumPropagation"}
        packed = make_node(
            node.op,
            f"{node.id}.packed",
            tuple(module.node_map[name] for name in node.inputs),
            {**dict(node.attrs), "output_lanes": (*lanes, *node.attrs.get("output_lanes", ()))},
            metadata,
        )
        restored = make_node(
            "tensors.unpack", node.id, (packed,), {"axes": (tensor_of(node.type).rank - 1,) * len(lanes)}, metadata
        )
        assert restored.type == node.type
        return RewriteResult(restored, (packed,))

    return RewriteRule(
        "VectorizeReduceSumPropagation",
        lambda node, _: node.op == "math.reduce_sum",
        rewrite,
        root_ops=("math.reduce_sum",),
    )


__all__ = ["reduce_sum_producer_rule"]
