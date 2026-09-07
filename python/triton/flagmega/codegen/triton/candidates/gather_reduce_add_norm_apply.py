# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Triton candidates for fused partial residual normalization."""

from __future__ import annotations

from math import prod

from triton.flagmega.ir import DistributedType, ReduceOp, TupleType
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.ntt.matmul_norm_stats_combine import (
    can_materialize_matmul_partial,
)

from .core import TritonCandidateContext, TritonCandidateProposal


class GatherReduceAddNormApplyCandidateProvider:
    """Bind a target-neutral fused collective to reviewed implementations."""

    op_names = frozenset({"ntt.gather_reduce_add_norm_apply"})

    def propose(
        self,
        node,
        context: TritonCandidateContext,
    ) -> TritonCandidateProposal | None:
        if len(node.inputs) != 4 or not isinstance(node.type, TupleType):
            return None
        if len(node.type.fields) != 2:
            return None
        source_type = context.module.node_map[node.inputs[0]].type
        addend_type = context.module.node_map[node.inputs[1]].type
        value_type, norm_output_type = node.type.fields
        if (
            not isinstance(source_type, DistributedType)
            or source_type.partial is None
            or source_type.partial.reduce_op is not ReduceOp.SUM
            or not source_type.partial.axes
            or addend_type != value_type
            or not can_materialize_matmul_partial(source_type, value_type)
            or not isinstance(value_type, DistributedType)
            or norm_output_type != value_type
        ):
            return None
        value = tensor_of(value_type)
        axis = int(node.attrs["axis"])
        axis = axis + value.rank if axis < 0 else axis
        if axis < 0 or axis >= value.rank:
            return None
        if any(not dimension.is_fixed for dimension in value.shape):
            return None
        outer_rows = prod(
            (dimension.fixed_value for dimension in value.shape[:axis]),
            start=1,
        )
        if outer_rows != 1:
            # This implementation owns one decode row.  Batched prefill uses
            # a separately reviewable row-indexed workspace/launch contract.
            return None
        placement = value_type.placement
        partial_axes = tuple(source_type.partial.axes)
        owner_count = prod(placement.hierarchy)
        partial_owner_count = prod(
            placement.hierarchy[index] for index in partial_axes
        )
        implementations = context.implementations(
            "gather_reduce_add_norm_apply",
            reduction="sum",
            axis_kind="suffix",
            supports_mean=True,
            outer_rows="single",
        )
        implementations = tuple(
            implementation
            for implementation in implementations
            if context.cooperative_grid
            or "cooperative_grid" not in implementation.requires
        )
        candidates = tuple(
            context.configure_implementation(
                implementation,
                semantic_parameters={
                    "source_type": source_type,
                    "value_type": value_type,
                    "norm_output_type": norm_output_type,
                    "partial_axes": partial_axes,
                    "partial_owner_count": partial_owner_count,
                    "owner_count": owner_count,
                },
                facts={
                    "collective_semantics": "gather-reduce-add-private-stats-norm-apply",
                    "private_norm_stats_workspace": True,
                    "placement_owner_count": owner_count,
                },
            )
            for implementation in implementations
        )
        if not candidates:
            return None
        return TritonCandidateProposal(
            candidates,
            context.choose_default("gather_reduce_add_norm_apply", candidates),
        )


__all__ = ["GatherReduceAddNormApplyCandidateProvider"]
