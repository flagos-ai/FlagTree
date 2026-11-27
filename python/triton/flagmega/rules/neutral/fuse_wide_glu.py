# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fuse explicit BF16 projections/FP32 SiLU/product/BF16 result boundaries."""

from triton.flagmega.ir import DType, TensorType
from triton.flagmega.pattern_match import F, is_alt, wildcard
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.neutral._utility import make_node


def fuse_wide_glu_rule():
    value = wildcard("value")
    gate = F.math.is_matmul(value, wildcard("gate_weight"), transpose_a=False, transpose_b=True, call_name="gate")
    up = F.math.is_matmul(value, wildcard("up_weight"), transpose_a=False, transpose_b=True, call_name="up")
    activation = F.math.is_silu(F.tensors.is_cast(gate, dtype="float32"), call_name="activation")
    up_cast = F.tensors.is_cast(up, dtype="float32")
    product = is_alt(F.math.is_mul(activation, up_cast), F.math.is_mul(up_cast, activation))
    pattern = F.tensors.is_cast(product, dtype="bfloat16", call_name="root")

    def rewrite(result, module):
        root, gate, up = (result[name] for name in ("root", "gate", "up"))
        if (not isinstance(gate.type, TensorType) or gate.type != up.type
                or gate.type.dtype != DType.BFLOAT16 or gate.type.rank != 2):
            return root
        order = gate.attrs.get("reduction_order", "relaxed")
        if result["activation"].attrs.get("math_mode", "fast") != "fast":
            # The GLU contract currently declares fast sigmoid arithmetic.
            # Preserve an explicit accurate SiLU instead of discarding it.
            return root
        if up.attrs.get("reduction_order", "relaxed") != order:
            return root
        inputs = tuple(result[name] for name in ("value", "gate_weight", "up_weight"))
        replacement = make_node("nn.dense_matmul_glu", root.id, inputs,
                                {"activation": "silu", "round_activation": False, "reduction_order": order},
                                {**root.metadata, "formed_by": "FuseWideGlu"})
        return RewriteResult(replacement)

    return RewriteRule("FuseWideGlu", pattern, rewrite)


__all__ = ["fuse_wide_glu_rule"]
