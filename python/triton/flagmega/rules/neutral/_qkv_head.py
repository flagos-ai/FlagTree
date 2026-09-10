# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Declarative Q/K normalization and rotary boundary patterns."""

from dataclasses import dataclass

from triton.flagmega.ir import DType, Node
from triton.flagmega.ir.ops.nn._norm import normalize_axis
from triton.flagmega.pattern_match import F, is_alt, wildcard


@dataclass(frozen=True)
class QKVHead:
    norm: Node
    rope: Node
    value: Node
    cosine: Node
    sine: Node
    round_intermediates: bool


def qkv_head_pattern(prefix):

    def head(wide):
        value = wildcard(f"{prefix}_value")
        cosine, sine = wildcard(f"{prefix}_cosine"), wildcard(f"{prefix}_sine")
        norm_input = F.tensors.is_cast(value, dtype="float32") if wide else value
        # RoPE already promotes its tables internally. The explicit widening
        # may disappear after Cast folding, without changing the wide head's
        # normalization/rotation contract. Match both ordinary IR forms.
        cos_input = is_alt(F.tensors.is_cast(cosine, dtype="float32"), cosine) if wide else cosine
        sin_input = is_alt(F.tensors.is_cast(sine, dtype="float32"), sine) if wide else sine
        stats = F.nn.is_norm_stats(norm_input, call_name=f"{prefix}_stats")
        norm = F.nn.is_norm_apply(norm_input, stats, call_name=f"{prefix}_norm").with_user_count(1)
        rope = F.nn.is_rope(norm, cos_input, sin_input, call_name=f"{prefix}_rope").with_user_count(1)
        return (F.tensors.is_cast(rope, dtype="bfloat16", call_name=f"{prefix}_wide").with_user_count(1)
                if wide else rope)

    return is_alt(head(False), head(True), name=f"{prefix}_root")


def qkv_head_from_match(result, prefix):
    norm, stats, rope = (result[f"{prefix}_{part}"] for part in ("norm", "stats", "rope"))
    value, cosine, sine = (result[f"{prefix}_{part}"] for part in ("value", "cosine", "sine"))
    wide = result.get_value_or_default(f"{prefix}_wide") is not None
    value_type = getattr(value.type, "tensor", value.type)
    rank = value_type.rank
    # Structure and private users are proven by Pattern. Only attribute/type
    # relations and the numerical boundary contract remain here.
    if (bool(stats.attrs["use_mean"]) != bool(norm.attrs["use_mean"])
            or normalize_axis(int(stats.attrs["axis"]), rank) != normalize_axis(int(norm.attrs["axis"]), rank)):
        return None
    expected_dtype = DType.FLOAT32 if wide else value_type.dtype
    if getattr(norm.type, "tensor", norm.type).dtype != expected_dtype:
        return None
    if wide:
        if value_type.dtype != DType.BFLOAT16:
            return None
        if any(getattr(node.type, "tensor", node.type).dtype not in {DType.BFLOAT16, DType.FLOAT32}
               for node in (cosine, sine)):
            return None
    return QKVHead(norm, rope, value, cosine, sine, not wide)


def same_table(lhs: Node, rhs: Node) -> bool:
    return lhs.id == rhs.id or (lhs.op == rhs.op == "tensors.cast" and
                                (lhs.inputs, lhs.attrs, lhs.type) == (rhs.inputs, rhs.attrs, rhs.type))
