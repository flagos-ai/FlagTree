# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Microkernel selection for partial-statistics normalization apply."""

from __future__ import annotations

from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import DistributedType, ReduceOp, SBPBroadCast, type_from_data

from .core import TIRMicroKernelContext, TIRMicroKernelProposal


class GatherReduceNormApplyMicroKernelProvider:
    op_names = frozenset({"ntt.gather_reduce_norm_apply"})
    family = "gather_reduce_norm_apply"

    def propose(
        self, context: TIRMicroKernelContext
    ) -> TIRMicroKernelProposal | None:
        dispatch = context.dispatch
        if dispatch.semantic_op not in self.op_names:
            return None
        if len(dispatch.arguments) != 4 or len(dispatch.outputs) != 1:
            raise CodegenError(
                "GatherReduceNormApply semantic TIR requires four arguments "
                f"and one output, got {len(dispatch.arguments)} and "
                f"{len(dispatch.outputs)}."
            )
        partial = context.function.parameter_map[dispatch.arguments[0]].type
        materialized = dispatch.semantic_attrs.get("materialized_stats_type")
        if isinstance(materialized, dict):
            materialized = type_from_data(materialized)
        if (
            not isinstance(partial, DistributedType)
            or not isinstance(materialized, DistributedType)
            or partial.partial is None
            or partial.partial.reduce_op is not ReduceOp.SUM
            or not partial.partial.axes
            or materialized.partial is not None
            or partial.tensor != materialized.tensor
            or partial.placement != materialized.placement
            or partial.axis_policies != materialized.axis_policies
            or any(
                not isinstance(policy, SBPBroadCast)
                for policy in partial.axis_policies
            )
        ):
            raise CodegenError(
                "GatherReduceNormApply requires compatible Sum-partial and "
                "materialized broadcast statistics."
            )
        implementations = context.implementations(
            self.family,
            supports_mean=True,
            axis_kind="suffix",
        )
        if not implementations:
            raise CodegenError(
                f"Implementation model {context.implementation_model.name!r} "
                "has no GatherReduceNormApply implementation."
            )
        candidates = tuple(context.candidate(value) for value in implementations)
        return TIRMicroKernelProposal(
            candidates,
            context.choose_default(self.family, candidates),
        )


__all__ = ["GatherReduceNormApplyMicroKernelProvider"]
