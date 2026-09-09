# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fuse a private normalization's final conversion, preserving input rounding."""

from triton.flagmega.ir import DType, TensorType
from triton.flagmega.pattern_match import F
from triton.flagmega.rules import RewriteResult, RewriteRule
from ._utility import make_node


def fuse_norm_apply_cast_rule():
    norm = F.nn.is_norm_apply(call_name="norm").with_user_count(1)
    pattern = F.tensors.is_cast(norm, call_name="root")

    def rewrite(result, module):
        root, norm = result["root"], result["norm"]
        if (not isinstance(root.type, TensorType) or root.type.dtype not in {DType.BFLOAT16, DType.FLOAT32}
                or norm.attrs.get("output_dtype") is not None):
            return root
        replacement = make_node("nn.norm_apply", root.id,
                                tuple(module.node_map[value] for value in norm.inputs),
                                {**norm.attrs, "output_dtype": root.type.dtype.value},
                                {**root.metadata, "formed_by": "FuseNormApplyCast"})
        return RewriteResult(replacement)

    return RewriteRule("FuseNormApplyCast", pattern, rewrite)


__all__ = ["fuse_norm_apply_cast_rule"]
