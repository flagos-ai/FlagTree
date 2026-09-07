# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Move Pack through a transpose while remapping its logical axes."""

from triton.flagmega.ir import IRModule, Node, get_definition
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.tensors.pack import normalize_axes
from triton.flagmega.pattern_match import F, MatchResult, wildcard
from triton.flagmega.rules import RewriteResult, RewriteRule


_INPUT = wildcard("input")
_PATTERN = F.tensors.is_pack(
    F.tensors.is_permute(_INPUT, call_name="transpose"),
    call_name="pack",
)


def _rewrite(result: MatchResult, _module: IRModule) -> RewriteResult:
    pack = result["pack"]
    transpose = result["transpose"]
    source = result["input"]
    assert all(isinstance(value, Node) for value in (pack, transpose, source))
    lanes = tuple(int(value) for value in pack.attrs["lanes"])
    raw_axes = (
        tuple(int(value) for value in pack.attrs["axes"])
        if "axes" in pack.attrs
        else (int(pack.attrs["axis"]),) * len(lanes)
    )
    axes = normalize_axes(raw_axes, tensor_of(transpose.type).rank)
    permutation = tuple(int(value) for value in transpose.attrs["axes"])
    input_axes = tuple(permutation[axis] for axis in axes)

    pack_definition = get_definition("tensors.pack")
    prepared_pack = pack_definition.prepare(
        (source,), {"lanes": lanes, "axes": input_axes}
    )
    packed = Node(
        f"{pack.id}.fold_pack_transpose.pack",
        "tensors.pack",
        tuple(value.id for value in prepared_pack.inputs),
        prepared_pack.result_type,
        prepared_pack.effect,
        prepared_pack.attrs,
        {"rewritten_by": "FoldPackTranspose"},
    )
    transpose_definition = get_definition("tensors.permute")
    prepared_transpose = transpose_definition.prepare((packed,), transpose.attrs)
    if prepared_transpose.result_type != pack.type:
        # This should follow from Pack/Permute type inference, but refusing an
        # equality is safer than admitting a malformed editable checkpoint.
        return RewriteResult(pack)
    replacement = Node(
        pack.id,
        "tensors.permute",
        (packed.id,),
        prepared_transpose.result_type,
        prepared_transpose.effect,
        prepared_transpose.attrs,
        {**dict(pack.metadata), "rewritten_by": "FoldPackTranspose"},
    )
    return RewriteResult(replacement, (packed,))


def fold_pack_transpose_rule() -> RewriteRule:
    return RewriteRule("FoldPackTranspose", pattern=_PATTERN, rewrite=_rewrite)


__all__ = ["fold_pack_transpose_rule"]
