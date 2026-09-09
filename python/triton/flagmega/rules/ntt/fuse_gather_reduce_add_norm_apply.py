# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fuse residual materialization/statistics with its sole NormApply consumer."""

from __future__ import annotations

from dataclasses import replace

from triton.flagmega.rules.core import RewriteResult, RewriteRule, RewriteEffectPolicy
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import DistributedType, IRModule, Node, ReduceOp, TupleType
from triton.flagmega.pattern_match import F, is_alt, wildcard
from triton.flagmega.ir.ops.builtin.get_item import GetItem
from triton.flagmega.ir.ops.ntt.gather_reduce_add_norm_apply import (
    GatherReduceAddNormApply, )
from triton.flagmega.rules.ntt.fuse_gather_reduce_norm_apply import (
    _is_zero_splat, )


def fuse_gather_reduce_add_norm_apply_rule() -> RewriteRule:
    """Fuse one private materialization and its consumer; shared uses stay intact."""
    combine = F.ntt.is_matmul_norm_stats_combine(call_name="combine")
    value = F.tensors.is_get_item(combine, 0, call_name="value_projection")
    stats = F.tensors.is_get_item(combine, 1, call_name="stats_projection").with_user_count(1)
    value_input = is_alt(value, F.distributed.is_sharded_view(value, call_name="value_view").with_user_count(1))
    stats_input = is_alt(stats, F.distributed.is_boxing(stats, call_name="stats_boxing").with_user_count(1))
    return RewriteRule(
        "fuse_gather_reduce_add_norm_apply",
        F.nn.is_norm_apply(value_input, stats_input, wildcard("scale"), wildcard("bias"), call_name="norm"),
        _rewrite,
        effect_policy=RewriteEffectPolicy.ALLOW,
    )


def _rewrite(result, module: IRModule):
    norm = result["norm"]
    node_map = module.node_map
    users = _users(module)
    positions = {node.id: index for index, node in enumerate(module.nodes)}
    value_view = result.get_value_or_default("value_view")
    stats_boxing = result.get_value_or_default("stats_boxing")
    value_projection, stats_projection, combine = (result[name]
                                                   for name in ("value_projection", "stats_projection", "combine"))
    combine_users = users.get(combine.id, ())
    if any(
            user.startswith("@") or not _projection_from(node_map.get(user), combine.id, {0, 1})
            for user in combine_users):
        return norm
    stats_projections = tuple(node_map[user] for user in combine_users if int(node_map[user].attrs["index"]) == 1)
    if stats_projections != (stats_projection, ):
        return norm
    value_projections = tuple(node_map[user] for user in combine_users if int(node_map[user].attrs["index"]) == 0)
    if not value_projections:
        return norm
    if value_view is not None:
        if (value_projections != (value_projection, ) or users.get(value_projection.id) != (value_view.id, )):
            # Refining the fused materialization changes the first result
            # distribution.  It is legal only when the coarse projection
            # exists solely to feed this private view.
            return norm
    elif any(user != norm.id and not user.startswith("@") and positions[user] < positions[norm.id]
             for projection in value_projections
             for user in users.get(projection.id, ())):
        # Moving the value publication to the consumer position may not
        # cross an earlier physical use.
        return norm
    source = node_map[combine.inputs[0]]
    if (not isinstance(source.type, DistributedType) or source.type.partial is None
            or source.type.partial.reduce_op is not ReduceOp.SUM or not source.type.partial.axes):
        return norm
    scale = node_map[norm.inputs[2]]
    bias = node_map[norm.inputs[3]]
    if positions[scale.id] >= positions[norm.id] or positions[bias.id] >= positions[norm.id]:
        return norm
    addend = node_map[combine.inputs[1]]
    private_addend_view: Node | None = None
    rewound_addend_view: Node | None = None
    if value_view is None or addend.type == value_view.type:
        fused_addend = addend
    elif (addend.op == "distributed.sharded_view" and len(addend.inputs) == 1
          and node_map[addend.inputs[0]].type == value_view.type and users.get(addend.id) == (combine.id, )):
        # The common down-projection pattern first widens a fine residual
        # view for the coarse combine and narrows the result again for
        # normalization.  Elementwise addition commutes with these views,
        # so keep the original fine value and remove both inverse aliases.
        fused_addend = node_map[addend.inputs[0]]
        rewound_addend_view = addend
    else:
        # Preserve the ShardedView as a real input alias.  Bufferization
        # can then enforce canonical backing when the refinement is not a
        # local subview instead of losing the physical contract in pass
        # metadata.
        private_addend_view = replace(
            value_view,
            inputs=(addend.id, ),
            metadata={
                **dict(value_view.metadata),
                "rewritten_from_result_view": value_projection.id,
                "fused_by": "FuseGatherReduceAddNormApply",
            },
        )
        fused_addend = private_addend_view
    inputs = (source, fused_addend, scale, bias)
    attrs = {
        "axis":
        int(norm.attrs["axis"]),
        "epsilon":
        float(norm.attrs["epsilon"]),
        "use_mean":
        bool(norm.attrs["use_mean"]),
        "round_before_scale":
        bool(norm.attrs.get("round_before_scale", False)),
        **({"output_dtype": norm.attrs["output_dtype"]} if "output_dtype" in norm.attrs else {}),
        "has_bias":
        not _is_zero_splat(bias, module),
    }
    try:
        result_type = GatherReduceAddNormApply.infer_call_type(inputs, attrs)
        effect = GatherReduceAddNormApply.infer_effect(inputs, attrs)
    except (IRSchemaError, TypeError, ValueError, KeyError):
        return norm
    if (not isinstance(result_type, TupleType) or result_type.fields != (
            value_view.type if value_view is not None else combine.type.fields[0],
            norm.type,
    )):
        return norm
    fused = replace(
        combine,
        op=GatherReduceAddNormApply.op_name,
        inputs=tuple(value.id for value in inputs),
        type=result_type,
        effect=effect,
        attrs=attrs,
        metadata={
            **dict(combine.metadata),
            "fused_by":
            "FuseGatherReduceAddNormApply",
            "fused_norm_apply":
            norm.id,
            "private_norm_stats":
            stats_projection.id,
            "private_value_view": (None if value_view is None else value_view.id),
            "private_addend_view": (private_addend_view.id if private_addend_view is not None else
                                    rewound_addend_view.id if rewound_addend_view is not None else None),
            "private_stats_boxing": (None if stats_boxing is None else stats_boxing.id),
            "norm_vectorization": {
                key: value
                for key, value in norm.metadata.items()
                if key in {
                    "selected_vectorization",
                    "selected_vector_axes",
                    "selected_vector_lanes",
                    "vectorization_candidate",
                    "vector_axes",
                    "vector_lanes",
                }
            },
        },
    )
    norm_projection = replace(
        norm,
        op=GetItem.op_name,
        inputs=(combine.id, ),
        effect=GetItem.infer_effect((fused, ), {"index": 1}),
        attrs={"index": 1},
        metadata={
            **dict(norm.metadata),
            "fused_source": combine.id,
            "fused_by": "FuseGatherReduceAddNormApply",
        },
    )
    prefix = (
        *((private_addend_view, ) if private_addend_view is not None else ()),
        fused,
        *(value_projections if value_view is None else ()),
    )
    removed = (
        combine.id,
        *(projection.id for projection in value_projections),
        stats_projection.id,
        *((value_view.id, ) if value_view is not None else ()),
        *((rewound_addend_view.id, ) if rewound_addend_view is not None else ()),
        *((stats_boxing.id, ) if stats_boxing is not None else ()),
    )
    return RewriteResult(norm_projection, prefix, removed_ids=removed)


def _projection_from(node: Node | None, source_id: str, indices: set[int]) -> bool:
    return (node is not None and node.op == GetItem.op_name and node.inputs == (source_id, )
            and node.attrs.get("index") in indices)


def _users(module: IRModule) -> dict[str, tuple[str, ...]]:
    result: dict[str, list[str]] = {node.id: [] for node in module.nodes}
    for node in module.nodes:
        for input_id in node.inputs:
            result.setdefault(input_id, []).append(node.id)
    for function in module.functions:
        for output in function.outputs:
            result.setdefault(output, []).append(f"@{function.name}:return")
    return {key: tuple(value) for key, value in result.items()}


__all__ = ["fuse_gather_reduce_add_norm_apply_rule"]
