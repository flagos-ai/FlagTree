# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nncase-shaped AutoDistributed search/extract/materialize pass."""

from __future__ import annotations

from dataclasses import replace

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import IRModule, Placement, verify_module
from triton.flagmega.passes.auto_distributed.candidates import DistributedCandidateProviderRegistry
from triton.flagmega.passes.auto_distributed.materializer import (
    DistributedMaterializer,
    distribution_selection_state,
)
from triton.flagmega.passes.auto_distributed.search import build_search_graph, solve_search_graph


class AutoDistributedPass:
    name = "AutoDistributed"
    preserves = frozenset()

    @classmethod
    def propose(cls, module: IRModule, target) -> IRModule:
        """Solve defaults but keep the logical graph editable and unmaterialized."""

        graph = cls._build_graph(module, target)
        result = solve_search_graph(graph, dump_subdirectory="Proposal")
        points, records = distribution_selection_state(
            result,
            policy=target.distribution_policy.identity,
        )
        old_ids = {
            point.id for point in module.selection_points
            if point.kind == "distribution"
        }
        metadata = dict(module.metadata)
        metadata["auto_distribution_proposal"] = {
            "schema": "flagmega.auto-distributed-proposal/v1",
            "placement": graph.placement.to_data(),
            "solver": "ortools-cp-sat",
            "status": result.status,
            "objective": result.objective,
        }
        proposed = replace(
            module,
            metadata=metadata,
            selection_points=tuple(
                point for point in module.selection_points
                if point.id not in old_ids
            ) + points,
            selections=tuple(
                record for record in module.selections
                if record.point_id not in old_ids
            ) + records,
        )
        return verify_module(proposed)

    @classmethod
    def apply(cls, module: IRModule, target) -> IRModule:
        """Constrain CP-SAT from editable selections and materialize one graph."""

        graph = cls._build_graph(module, target)
        proposal_points = {
            point.id: point
            for point in module.selection_points
            if point.kind == "distribution"
        }
        proposal_records = {
            point_id: module.selection_map[point_id]
            for point_id in proposal_points
            if point_id in module.selection_map
        }
        expected = {
            f"distribution.{bucket.node_id}": bucket.node_id
            for bucket in graph.buckets
            if bucket.executable and len(bucket.candidates) > 1
        }
        missing = sorted(set(expected) - set(proposal_points))
        missing_records = sorted(set(expected) - set(proposal_records))
        unexpected = sorted(set(proposal_points) - set(expected))
        if missing or missing_records or unexpected:
            raise IRVerificationError(
                "AutoDistributed apply requires the exact proposal selection "
                f"surface (missing_points={missing}, "
                f"missing_records={missing_records}, unexpected={unexpected}).",
                stage=module.stage,
            )
        fixed = {
            expected[point_id]: record.candidate_id
            for point_id, record in proposal_records.items()
        }
        result = solve_search_graph(
            graph,
            fixed_selections=fixed,
            dump_subdirectory="Applied",
        )
        materialized = DistributedMaterializer(
            result,
            policy=target.distribution_policy.identity,
            proposal_points=proposal_points,
            proposal_records=proposal_records,
        ).run()
        return verify_module(materialized)

    @classmethod
    def run(cls, module: IRModule, target) -> IRModule:
        """Standalone default compilation: propose, then apply without a pause."""

        if any(point.kind == "distribution" for point in module.selection_points):
            return cls.apply(module, target)
        return cls.apply(cls.propose(module, target), target)

    @staticmethod
    def _build_graph(module: IRModule, target):
        placements = tuple(target.distributed_placements(module))
        if not placements:
            raise ValueError(f"Target {target.name!r} did not provide an AutoDistributed placement.")
        if len(placements) != 1:
            raise ValueError(
                "AutoDistributed requires exactly one selected Placement per "
                f"search, but target {target.name!r} provided {len(placements)}. "
                "Topology alternatives must be selected before distribution "
                "search rather than being silently ordered."
            )
        registry = DistributedCandidateProviderRegistry()
        target.register_auto_distributed_candidate_providers(registry)
        realization_policy = target.distributed_reshard_realization_policy()
        return build_search_graph(
            module,
            placements[0],
            registry,
            realization_policy,
            target.distributed_reshard_cost_model(),
            target.distributed_operation_cost_model(),
        )


__all__ = ["AutoDistributedPass"]
