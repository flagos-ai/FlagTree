# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Post-AutoDistribution lowering for the packed Q/K/V combine boundary."""

from __future__ import annotations

from triton.flagmega.ir import DistributedType, IRModule, Node, TupleType
from triton.flagmega.ir.ops.distributed.boxing import Boxing
from triton.flagmega.pattern_match import F, wildcard
from triton.flagmega.rules import RewriteRedirect, RewriteRule


def fold_materialized_packed_qkv_parallel_linear_combine_rule() -> RewriteRule:
    """Port nncase's exact materialized-identity fold."""

    pattern = F.ntt.is_packed_qkv_parallel_linear_combine(
        wildcard("qkv"),
        target_name="target",
        call_name="call",
    )

    def rewrite(result, module: IRModule):
        del module
        call = result["call"]
        qkv = result["qkv"]
        assert isinstance(call, Node) and isinstance(qkv, Node)
        if qkv.type != call.type or not _is_materialized_tuple(qkv.type):
            return call
        return RewriteRedirect(qkv.id)

    return RewriteRule(
        "FoldMaterializedPackedQKVParallelLinearCombine",
        pattern,
        rewrite,
    )


def lower_packed_qkv_parallel_linear_combine_rule() -> RewriteRule:
    """Port nncase's remaining partial combine to generic tuple Boxing."""

    pattern = F.ntt.is_packed_qkv_parallel_linear_combine(
        wildcard("qkv"),
        target_name="target",
        call_name="call",
    )

    def rewrite(result, module: IRModule):
        call = result["call"]
        qkv = result["qkv"]
        assert isinstance(call, Node) and isinstance(qkv, Node)
        if not _is_three_field_partial_tuple(qkv.type):
            return call
        prepared = Boxing.prepare((qkv,), {"new_type": call.type})
        return Node(
            call.id,
            Boxing.op_name,
            tuple(value.id for value in prepared.inputs),
            prepared.result_type,
            prepared.effect,
            prepared.attrs,
            {
                **dict(call.metadata),
                "lowered_by": "LowerPackedQKVParallelLinearCombine",
            },
        )

    return RewriteRule(
        "LowerPackedQKVParallelLinearCombine",
        pattern,
        rewrite,
    )


def _is_materialized_tuple(value_type) -> bool:
    return isinstance(value_type, TupleType) and all(
        not isinstance(field, DistributedType) or field.partial is None
        for field in value_type.fields
    )


def _is_three_field_partial_tuple(value_type) -> bool:
    return (
        isinstance(value_type, TupleType)
        and len(value_type.fields) == 3
        and all(
            isinstance(field, DistributedType) and field.partial is not None
            for field in value_type.fields
        )
    )


__all__ = [
    "fold_materialized_packed_qkv_parallel_linear_combine_rule",
    "lower_packed_qkv_parallel_linear_combine_rule",
]
