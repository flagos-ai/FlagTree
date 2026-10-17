# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Push final-axis demands into projections with an output_lanes contract."""

from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.neutral._utility import make_node
from triton.flagmega.rules.ntt.vectorize.propagation.producer_demand import final_axis_pack_demand


def block_scaled_matmul_producer_rule():
    def rewrite(node, module):
        lanes = final_axis_pack_demand(module, {node.id})
        if lanes is None:
            return None
        metadata = {"rewritten_by": "VectorizeBlockScaledMatMulPropagation"}
        packed = make_node(
            node.op,
            f"{node.id}.packed",
            tuple(module.node_map[name] for name in node.inputs),
            {**dict(node.attrs), "output_lanes": (*lanes, *node.attrs.get("output_lanes", ()))},
            {**dict(node.metadata), **metadata},
        )
        restored = make_node(
            "tensors.unpack",
            node.id,
            (packed,),
            {"axes": (tensor_of(node.type).rank - 1,) * len(lanes)},
            {**dict(node.metadata), **metadata},
        )
        assert restored.type == node.type
        return RewriteResult(restored, (packed,))

    return RewriteRule(
        "VectorizeBlockScaledMatMulPropagation",
        lambda node, _: node.op in {"math.block_scaled_matmul", "math.packed_block_scaled_matmul",
                                    "nn.partitioned_sparse_experts_down", "nn.mla_projection"},
        rewrite,
        root_ops=("math.block_scaled_matmul", "math.packed_block_scaled_matmul", "nn.partitioned_sparse_experts_down",
                  "nn.mla_projection"),
    )


__all__ = ["block_scaled_matmul_producer_rule"]
