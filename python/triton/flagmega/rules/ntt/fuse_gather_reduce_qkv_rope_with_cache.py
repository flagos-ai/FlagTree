# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fuse private partial-QKV materialization using shared-producer patterns."""

from dataclasses import replace

from triton.flagmega.ir.ops.ntt.gather_reduce_qkv_rope_with_cache import GatherReduceQKVRoPEWithCache
from triton.flagmega.ir.ops.ntt.packed_qkv_parallel_linear_combine import can_materialize_packed_qkv
from triton.flagmega.pattern_match import F, is_alt, is_op, is_unary_chain, wildcard
from triton.flagmega.rules.core import RewriteResult, RewriteRule


def fuse_gather_reduce_qkv_rope_with_cache_rule() -> RewriteRule:
    source = wildcard("source")
    combine = is_alt(
        F.distributed.is_boxing(source),
        F.ntt.is_packed_qkv_parallel_linear_combine(source),
        name="combine",
    ).with_user_count(3)
    step = is_alt(*(is_op(op) for op in (
        "builtin.identity",
        "distributed.sharded_view",
        "tensors.reshape",
    ))).with_user_count(1)
    fields = tuple(
        is_unary_chain(
            F.tensors.is_get_item(combine, index, call_name=f"projection_{index}").with_user_count(1),
            step,
            name=f"views_{index}",
        ) for index in range(3))
    qkv = F.builtin.is_tuple(*fields, call_name="qkv").with_user_count(1)
    return RewriteRule("fuse_gather_reduce_qkv_rope_with_cache", F.nn.is_qkv_rope_with_cache(qkv, call_name="consumer"),
                       _rewrite)


def _rewrite(result, module):
    consumer, source, combine, qkv = (result[name] for name in ("consumer", "source", "combine", "qkv"))
    if not can_materialize_packed_qkv(source.type, combine.type):
        return consumer
    attrs = {
        **dict(consumer.attrs),
        "materialized_qkv_type": combine.type,
        "logical_qkv_type": qkv.type,
    }
    inputs = (source, *(module.node_map[value] for value in consumer.inputs[1:]))
    try:
        inferred = GatherReduceQKVRoPEWithCache.infer_call_type(inputs, attrs)
    except (TypeError, ValueError, KeyError):
        return consumer
    if inferred != consumer.type:
        return consumer
    removed = {qkv.id, combine.id}
    for index in range(3):
        removed.add(result[f"projection_{index}"].id)
        removed.update(node.id for node in result[f"views_{index}"])
    replacement = replace(
        consumer,
        op=GatherReduceQKVRoPEWithCache.op_name,
        inputs=tuple(node.id for node in inputs),
        attrs=attrs,
        metadata={**dict(consumer.metadata), "fused_partial_qkv_materialization": sorted(removed)},
    )
    return RewriteResult(replacement, removed_ids=tuple(sorted(removed)))


__all__ = ["fuse_gather_reduce_qkv_rope_with_cache_rule"]
