# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Post-AutoDistribution lowering for the packed Q/K/V combine boundary."""

from __future__ import annotations

from dataclasses import replace

from triton.flagmega.ir import DistributedType, IRModule, Node, TupleType
from triton.flagmega.ir.ops.distributed.boxing import Boxing
from triton.flagmega.pattern_match import F, wildcard
from triton.flagmega.rules import RewriteRedirect, RewriteResult, RewriteRule
from triton.flagmega.rules.neutral._utility import make_node


def fold_materialized_packed_qkv_parallel_linear_combine_rule() -> RewriteRule:
    """Port the reference exact materialized-identity fold."""

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


def lower_packed_qkv_parallel_linear_combine_rule(*, materialize_rounding=False) -> RewriteRule:
    """Port the reference remaining partial combine to generic tuple Boxing."""

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
        if any(source.tensor.dtype != target.tensor.dtype for source, target in zip(qkv.type.fields, call.type.fields)):
            # Retain the semantic rounding boundary until post-distribution
            # fusion has had its opportunity to consume the complete region.
            if not materialize_rounding:
                return call
            # A typed combine can round AFTER the complete owner sum. Keep
            # generic Boxing dtype-preserving and express each result cast.
            wide_type = TupleType(
                tuple(
                    replace(target, tensor=replace(target.tensor, dtype=source.tensor.dtype))
                    for source, target in zip(qkv.type.fields, call.type.fields)))
            used = set(module.node_map)

            def fresh(base):
                name, index = base, 0
                while name in used:
                    index += 1
                    name = f"{base}.{index}"
                used.add(name)
                return name

            wide = make_node("distributed.boxing", fresh(call.id + ".accumulator"), (qkv, ), {"new_type": wide_type},
                             call.metadata)
            prefix, results = [wide], []
            for index, target in enumerate(call.type.fields):
                field = make_node("builtin.get_item", fresh(call.id + f".field{index}"), (wide, ), {"index": index},
                                  call.metadata)
                dtype = target.tensor.dtype
                cast = make_node("ntt.vectorized_cast", fresh(call.id + f".rounded{index}"), (field, ),
                                 {"new_type": dtype, "vectorize_axes":
                                  (-1, ), "rounding_contract": "preserve"}, call.metadata)
                prefix.extend((field, cast))
                results.append(cast)
            result = make_node("builtin.tuple", call.id, tuple(results), {}, call.metadata)
            assert result.type == call.type
            return RewriteResult(result, prefix_nodes=tuple(prefix))
        prepared = Boxing.prepare((qkv, ), {"new_type": call.type})
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
    return isinstance(value_type, TupleType) and all(not isinstance(field, DistributedType) or field.partial is None
                                                     for field in value_type.fields)


def _is_three_field_partial_tuple(value_type) -> bool:
    return (isinstance(value_type, TupleType) and len(value_type.fields) == 3
            and all(isinstance(field, DistributedType) and field.partial is not None for field in value_type.fields))


__all__ = [
    "fold_materialized_packed_qkv_parallel_linear_combine_rule",
    "lower_packed_qkv_parallel_linear_combine_rule",
]
