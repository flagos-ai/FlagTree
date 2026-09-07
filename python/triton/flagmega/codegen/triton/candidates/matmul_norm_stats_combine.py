# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Target-neutral candidates for partial materialization plus NormStats."""

from __future__ import annotations

from math import prod

from triton.flagmega.ir import DistributedType, Node, ReduceOp, SBPBroadCast, TupleType
from triton.flagmega.ir.ops.ntt.matmul_norm_stats_combine import (
    can_materialize_matmul_partial,
)
from triton.flagmega.codegen.triton.vectorization import (
    configured_vector_schedule,
    consumed_vector_contracts,
    vectorization_contract,
)

from .core import TritonCandidateContext, TritonCandidateProposal


class MatMulNormStatsCombineCandidateProvider:
    """Lower the nncase-shaped combine to a catalog-provided collective.

    The provider describes only distributed semantics.  A target catalog owns
    the concrete tile, launch requirements, and implementation template.
    """

    op_names = frozenset({"ntt.matmul_norm_stats_combine"})

    def propose(
        self,
        node,
        context: TritonCandidateContext,
    ) -> TritonCandidateProposal | None:
        if len(node.inputs) != 2 or not isinstance(node.type, TupleType):
            return None
        if len(node.type.fields) != 2:
            return None
        source_type = context.module.node_map[node.inputs[0]].type
        addend_type = context.module.node_map[node.inputs[1]].type
        value_type, stats_type = node.type.fields
        if addend_type != value_type or not can_materialize_matmul_partial(
            source_type, value_type
        ):
            return None
        if not isinstance(value_type, DistributedType):
            return None
        if not isinstance(stats_type, DistributedType):
            return None
        value = value_type.tensor
        if any(not dimension.is_fixed for dimension in value.shape[:-1]):
            return None
        if prod(
            (dimension.fixed_value for dimension in value.shape[:-1]),
            start=1,
        ) != 1:
            # The current sum_rms implementation materializes one decode row.
            # A multi-row variant needs row-indexed statistics workspaces and
            # is a different reviewed implementation, not an implicit mode.
            return None
        placement = value_type.placement
        if isinstance(source_type, DistributedType) and source_type.partial is not None:
            if source_type.partial.reduce_op is not ReduceOp.SUM:
                return None
            family = "gather_reduce_add_norm_stats"
            contract = {"reduction": "sum"}
            partial_axes = tuple(source_type.partial.axes)
        elif source_type == value_type:
            family = "add_norm_stats"
            contract = {"input_kind": "materialized"}
            partial_axes = ()
        else:
            return None
        owner_count = prod(placement.hierarchy)
        partial_owner_count = prod(
            placement.hierarchy[axis] for axis in partial_axes
        )
        fused_requirements = {}
        residual_add = node.metadata.get("residual_add")
        norm_consumer = node.metadata.get("norm_consumer")
        if isinstance(residual_add, str) and isinstance(norm_consumer, str):
            fused_requirements = consumed_vector_contracts(
                context.module,
                {
                    "residual_add": (residual_add, "axes"),
                    "norm": (norm_consumer, "reduction_axis"),
                },
            )
            if fused_requirements is None:
                return None
        value_contract = vectorization_contract(Node(
            node.id,
            "math.add",
            node.inputs,
            value_type,
            node.effect,
            {},
            node.metadata,
        ))
        implementations = context.implementations(
            family,
            **contract,
            axis_kind="last",
            use_mean=bool(node.attrs["use_mean"]),
            output_layout=(
                "broadcast"
                if all(
                    isinstance(policy, SBPBroadCast)
                    for policy in value_type.axis_policies
                )
                else "distributed"
            ),
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
                    "stats_type": stats_type,
                    "axis": int(node.attrs["axis"]),
                    "use_mean": bool(node.attrs["use_mean"]),
                    "partial_axes": partial_axes,
                    "partial_owner_count": partial_owner_count,
                    "owner_count": owner_count,
                    "vector_schedule": configured_vector_schedule(
                        value_contract,
                        lowering="collective_output_tile",
                        tile=int(implementation.parameters["tile"]),
                    ),
                    "consumed_vector_contracts": fused_requirements,
                    **(
                        {
                            "residual_add": residual_add,
                            "norm_consumer": norm_consumer,
                        }
                        if fused_requirements else {}
                    ),
                },
                facts={
                    **(
                        {"collective_semantics": "gather-reduce-add-norm-stats"}
                        if partial_axes
                        else {"local_semantics": "add-norm-stats"}
                    ),
                    "explicit_norm_stats_result": True,
                    "placement_owner_count": owner_count,
                },
            )
            for implementation in implementations
        )
        if not candidates:
            return None
        return TritonCandidateProposal(
            candidates,
            context.choose_default(family, candidates),
        )


__all__ = ["MatMulNormStatsCombineCandidateProvider"]
