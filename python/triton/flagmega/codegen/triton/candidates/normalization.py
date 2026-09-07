# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Triton candidates for explicit additive normalization dataflow."""

from __future__ import annotations

from collections.abc import Mapping

from triton.flagmega.ir import Node
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.codegen.triton.vectorization import (
    configured_vector_schedule,
    vectorization_contract,
)

from .core import TritonCandidateContext, TritonCandidateProposal


class NormStatsCandidateProvider:
    op_names = frozenset({"nn.norm_stats"})

    def propose(self, node: Node, context: TritonCandidateContext) -> TritonCandidateProposal | None:
        value = tensor_of(context.module.node_map[node.inputs[0]].type)
        axis = _normalize_axis(int(node.attrs["axis"]), value.rank)
        use_mean = bool(node.attrs["use_mean"])
        candidates = tuple(
            context.configure_implementation(
                implementation,
                semantic_parameters={"axis": axis, "use_mean": use_mean},
                facts={
                    "additive_statistics": (
                        ("sum", "square_sum") if use_mean else ("square_sum",)
                    ),
                    "local_shard_reduction": True,
                },
            )
            for implementation in context.implementations(
                "norm_stats", supports_mean=True, axis_kind="suffix"
            )
            if context.cooperative_grid
            or "cooperative_grid" not in implementation.requires
        )
        if not candidates:
            return None
        return TritonCandidateProposal(
            candidates,
            context.choose_default("norm_stats", candidates),
        )


class NormApplyCandidateProvider:
    op_names = frozenset({"nn.norm_apply"})

    def propose(self, node: Node, context: TritonCandidateContext) -> TritonCandidateProposal | None:
        value = tensor_of(context.module.node_map[node.inputs[0]].type)
        axis = _normalize_axis(int(node.attrs["axis"]), value.rank)
        use_mean = bool(node.attrs["use_mean"])
        contract = vectorization_contract(node)
        candidates = tuple(
            context.configure_implementation(
                implementation,
                semantic_parameters={
                    "axis": axis,
                    "use_mean": use_mean,
                    "bias_mode": "tensor",
                    "vector_schedule": _vector_schedule(
                        contract, implementation.parameters),
                },
                facts={
                    "external_materialized_stats": True,
                    "local_shard_apply": True,
                },
            )
            for implementation in context.implementations(
                "norm_apply", supports_mean=True, axis_kind="suffix",
                bias_mode="tensor",
            )
            if context.cooperative_grid
            or "cooperative_grid" not in implementation.requires
        )
        if not candidates:
            return None
        return TritonCandidateProposal(
            candidates,
            context.choose_default("norm_apply", candidates),
        )


def _vector_schedule(contract: Mapping[str, object], parameters: Mapping[str, object]):
    return configured_vector_schedule(
        contract,
        lowering="local_elementwise",
        block_size=int(parameters["block_size"]),
    )


def _normalize_axis(axis: int, rank: int) -> int:
    return axis + rank if axis < 0 else axis


__all__ = ["NormApplyCandidateProvider", "NormStatsCandidateProvider"]
