# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Consume a converted normalization numerator directly from its producer."""

from triton.flagmega.ir import DType, VectorType
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.pattern_match import F, wildcard
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.neutral._utility import make_node


def fuse_norm_apply_input_cast_rule():
    # Absorb only a private converted value. Shared users must retain one
    # common rounding boundary, which a producer epilogue can still own.
    pattern = F.nn.is_norm_apply(
        F.tensors.is_cast(wildcard("value"), call_name="cast").with_user_count(1), call_name="norm")

    def rewrite(result, module):
        norm, converted, value = result["norm"], result["cast"], result["value"]
        source, target, output = (tensor_of(node.type) for node in (value, converted, norm))
        scalar = lambda dtype: dtype.elem_type if isinstance(dtype, VectorType) else dtype
        lanes = lambda dtype: dtype.lanes if isinstance(dtype, VectorType) else ()
        if (norm.attrs.get("input_dtype") is not None
                or scalar(source.dtype) not in {DType.BFLOAT16, DType.FLOAT32}
                or scalar(target.dtype) not in {DType.BFLOAT16, DType.FLOAT32}
                or source.shape != target.shape or lanes(source.dtype) != lanes(target.dtype)):
            return norm
        # An already explicit virtual dtype may represent another ordered
        # conversion; do not flatten conversion chains across their rounding.
        replacement = make_node("nn.norm_apply", norm.id,
            (value, *(module.node_map[name] for name in norm.inputs[1:])),
            {**norm.attrs, "input_dtype": scalar(target.dtype).value,
             "output_dtype": scalar(output.dtype).value},
            {**norm.metadata, "formed_by": "FuseNormApplyInputCast"})
        if replacement.type != norm.type:
            return norm
        return RewriteResult(replacement)

    return RewriteRule("FuseNormApplyInputCast", pattern, rewrite)


__all__ = ["fuse_norm_apply_input_cast_rule"]
