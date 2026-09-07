# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fuse a private normalization's final conversion, preserving input rounding."""

from triton.flagmega.ir import DType, TensorType
from triton.flagmega.rules import RewriteResult, RewriteRule
from ._utility import make_node


def fuse_norm_apply_cast_rule():
    def match(root, module):
        if root.op != "tensors.cast" or not isinstance(root.type, TensorType):
            return None
        norm = module.node_map[root.inputs[0]]
        if (norm.op != "nn.norm_apply" or not norm.effect.is_pure
                or root.type.dtype not in {DType.BFLOAT16, DType.FLOAT32}
                or norm.attrs.get("output_dtype") is not None):
            return None
        if any(norm.id in f.outputs for f in module.functions):
            return None
        users = [node for node in module.nodes if norm.id in node.inputs]
        return norm if len(users) == 1 and users[0].id == root.id else None

    def rewrite(root, module):
        norm = match(root, module)
        replacement = make_node("nn.norm_apply", root.id,
                                tuple(module.node_map[value] for value in norm.inputs),
                                {**norm.attrs, "output_dtype": root.type.dtype.value},
                                {**root.metadata, "formed_by": "FuseNormApplyCast"})
        return RewriteResult(replacement)

    return RewriteRule("FuseNormApplyCast", lambda node, module: match(node, module) is not None, rewrite)


__all__ = ["fuse_norm_apply_cast_rule"]
