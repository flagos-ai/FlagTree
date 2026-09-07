# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fuse explicit BF16 projections/FP32 SiLU/product/BF16 result boundaries."""

from triton.flagmega.ir import DType, TensorType
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.neutral._utility import make_node


def fuse_wide_glu_rule():
    def match(root, module):
        nodes = module.node_map
        if root.op != "tensors.cast" or not isinstance(root.type, TensorType) or root.type.dtype != DType.BFLOAT16:
            return None
        product = nodes[root.inputs[0]]
        if product.op != "math.mul" or product.type.dtype != DType.FLOAT32:
            return None
        activation, up_cast = (nodes[value] for value in product.inputs)
        if up_cast.op == "math.silu":
            activation, up_cast = up_cast, activation
        if activation.op != "math.silu" or up_cast.op != "tensors.cast":
            return None
        gate_cast = nodes[activation.inputs[0]]
        if (gate_cast.op != "tensors.cast" or gate_cast.type.dtype != DType.FLOAT32
                or up_cast.type.dtype != DType.FLOAT32):
            return None
        gate, up = nodes[gate_cast.inputs[0]], nodes[up_cast.inputs[0]]
        if (gate.op != "math.matmul" or up.op != "math.matmul" or gate.type != up.type
                or gate.type.dtype != DType.BFLOAT16 or gate.type.rank != 2 or gate.inputs[0] != up.inputs[0]
                or any(value.attrs.get("transpose_a", False) or not value.attrs.get("transpose_b", False)
                       for value in (gate, up))):
            return None
        return tuple(nodes[value] for value in (gate.inputs[0], gate.inputs[1], up.inputs[1]))

    def rewrite(root, module):
        inputs = match(root, module)
        replacement = make_node("nn.dense_matmul_glu", root.id, inputs,
                                {"activation": "silu", "round_activation": False},
                                {**root.metadata, "formed_by": "FuseWideGlu"})
        return RewriteResult(replacement)

    return RewriteRule("FuseWideGlu", matches=lambda node, module: match(node, module) is not None, rewrite=rewrite)


__all__ = ["fuse_wide_glu_rule"]
