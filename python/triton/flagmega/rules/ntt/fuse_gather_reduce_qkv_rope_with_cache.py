# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fuse proven private QKV statistics only after distribution has selected layouts."""

from dataclasses import replace
from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import DType, VectorType
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.ntt.gather_reduce_qkv_rope_with_cache import GatherReduceQKVRoPEWithCache
from triton.flagmega.pattern_match import F, wildcard
from triton.flagmega.rules.core import RewriteResult, RewriteRule
from triton.flagmega.rules.ntt.fuse_gather_reduce_norm_apply import _users


_VIEWS = frozenset({"builtin.identity", "distributed.sharded_view", "tensors.reshape", "tensors.bitcast"})


def _scalar_dtype(node):
    dtype = tensor_of(node.type).dtype
    return dtype.elem_type if isinstance(dtype, VectorType) else dtype


def _scalar_shape(node, lane_axis=-1):
    tensor = tensor_of(node.type)
    shape = list(tensor.shape)
    if isinstance(tensor.dtype, VectorType):
        shape[lane_axis] *= prod(tensor.dtype.lanes)
    return tuple(shape)


def _trace(node, module, removed, *, casts=False):
    while len(node.inputs) == 1:
        parent = module.node_map[node.inputs[0]]
        if node.op in _VIEWS:
            if _scalar_dtype(node) != _scalar_dtype(parent):
                break
        elif casts and node.op in {"tensors.cast", "ntt.vectorized_cast"}:
            # BF16 -> FP32 is exact. No narrowing or reordering may disappear.
            if _scalar_dtype(parent) is not DType.BFLOAT16 or _scalar_dtype(node) is not DType.FLOAT32:
                break
            if node.op == "ntt.vectorized_cast":
                rank = tensor_of(parent.type).rank
                if any((int(axis) + rank) % rank != rank - 1 for axis in node.attrs["vectorize_axes"]):
                    break
        else:
            break
        shapes = (_scalar_shape(node), _scalar_shape(parent))
        if any(not dim.is_fixed for shape in shapes for dim in shape):
            break
        if prod(d.fixed_value for d in shapes[0]) != prod(d.fixed_value for d in shapes[1]):
            break
        removed.add(node.id)
        node = parent
    return node


def fuse_gather_reduce_qkv_rope_with_cache_rule():
    return RewriteRule(
        "FuseGatherReduceQKVStatisticsRoPEWithCache",
        F.nn.is_qkv_rope_with_cache(wildcard("qkv"), call_name="consumer"), _rewrite,
    )


def _rewrite(match, module):
    consumer, qkv = match["consumer"], match["qkv"]
    if qkv.op != "builtin.tuple" or len(qkv.inputs) != 3:
        return consumer
    removed = {qkv.id}
    projections = []
    for index, value_id in enumerate(qkv.inputs):
        value = module.node_map[value_id]
        projection = _trace(value, module, removed)
        if projection.op != "builtin.get_item" or projection.attrs["index"] != index:
            return consumer
        projections.append(projection)
        removed.add(projection.id)
        if index < 2:
            role = ("q", "k")[index]
            stats = module.node_map[consumer.inputs[10 + index]]
            if stats.op != "nn.norm_stats" or stats.attrs["use_mean"] != consumer.attrs[f"{role}_use_mean"]:
                return consumer
            stats_input = module.node_map[stats.inputs[0]]
            rank = tensor_of(value.type).rank
            if (int(stats.attrs["axis"]) % rank != int(consumer.attrs[f"{role}_axis"]) % rank
                    or _scalar_shape(stats_input) != _scalar_shape(value, consumer.attrs["qkv_layout"].index("dim"))):
                return consumer
            if _trace(stats_input, module, removed, casts=True).id != projection.id:
                return consumer
            removed.add(stats.id)
    if len({value.inputs[0] for value in projections}) != 1:
        return consumer
    combine = module.node_map[projections[0].inputs[0]]
    if combine.op not in {"ntt.packed_qkv_parallel_linear_combine", "distributed.boxing"}:
        return consumer
    removed.add(combine.id)
    users = _users(module)
    if any(set(users[value]) - removed - {consumer.id} for value in removed):
        return consumer
    source = module.node_map[combine.inputs[0]]
    inputs = (source, *(module.node_map[value] for value in consumer.inputs[1:10]))
    attrs = {**dict(consumer.attrs), "materialized_qkv_type": combine.type, "logical_qkv_type": qkv.type}
    try:
        prepared = GatherReduceQKVRoPEWithCache.prepare(inputs, attrs)
    except (IRSchemaError, TypeError, ValueError):
        return consumer
    if prepared.result_type != consumer.type:
        return consumer
    return RewriteResult(replace(
        consumer, op=GatherReduceQKVRoPEWithCache.op_name, inputs=tuple(node.id for node in prepared.inputs),
        type=prepared.result_type, attrs=prepared.attrs, effect=prepared.effect,
        metadata={**dict(consumer.metadata), "fused_qkv_statistics_region": tuple(sorted(removed))},
    ), removed_ids=tuple(sorted(removed)))
