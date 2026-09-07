# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit provider registry for semantic-op to Triton TIR candidates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import Candidate, IRModule, Node
from triton.flagmega.codegen.triton.implementation import (
    TritonImplementation,
    TritonImplementationModel,
)
from triton.flagmega.passes.tir.norm_consumer import NormConsumerMatch
from triton.flagmega.passes.tir.projection_logits_argmax import (
    ProjectionLogitsArgmaxMatch,
)
from triton.flagmega.passes.tir.projection_residual_norm import (
    ProjectionResidualNormMatch,
)


@dataclass(frozen=True)
class TritonCandidateProposal:
    candidates: tuple[Candidate, ...]
    default_candidate: str
    selection_kind: str = "tir"

    def __post_init__(self) -> None:
        if not self.candidates:
            raise CodegenError("A Triton candidate proposal cannot be empty.")
        candidate_ids = tuple(candidate.id for candidate in self.candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise CodegenError(
                f"A Triton candidate proposal contains duplicate ids: {candidate_ids}."
            )
        if self.default_candidate not in candidate_ids:
            raise CodegenError(
                f"Triton default candidate {self.default_candidate!r} is not in "
                f"the proposal {candidate_ids}."
            )
        if self.selection_kind not in {"tir", "semantic_tir"}:
            raise CodegenError(
                f"Unknown Triton candidate selection kind {self.selection_kind!r}."
            )


@dataclass(frozen=True)
class TritonCandidateContext:
    module: IRModule
    target: object
    projection_norm_matches: Mapping[str, ProjectionResidualNormMatch]
    projection_logits_matches: Mapping[str, ProjectionLogitsArgmaxMatch]
    norm_consumer_matches: Mapping[str, NormConsumerMatch]
    reusable_node_ids: frozenset[str]
    cooperative_grid: bool
    implementation_model: TritonImplementationModel

    def is_reusable(self, node: Node) -> bool:
        return node.id in self.reusable_node_ids

    def implementations(
        self,
        family: str,
        **contract: object,
    ) -> tuple[TritonImplementation, ...]:
        return self.implementation_model.find(family, **contract)

    def configure_implementation(
        self,
        implementation: TritonImplementation,
        *,
        semantic_parameters: Mapping[str, object] | None = None,
        facts: Mapping[str, object] | None = None,
    ) -> Candidate:
        semantic = dict(semantic_parameters or {})
        conflicts = {
            name: (value, semantic[name])
            for name, value in implementation.parameters.items()
            if name in semantic and semantic[name] != value
        }
        if conflicts:
            raise CodegenError(
                f"Semantic candidate {implementation.id!r} conflicts with target-owned "
                f"implementation parameters: {conflicts}."
            )
        semantic_facts = dict(facts or {})
        implementation_facts = dict(implementation.facts)
        fact_conflicts = {
            name: (value, semantic_facts[name])
            for name, value in implementation_facts.items()
            if name in semantic_facts and semantic_facts[name] != value
        }
        if fact_conflicts:
            raise CodegenError(
                f"Semantic candidate {implementation.id!r} conflicts with target-owned "
                f"implementation facts: {fact_conflicts}."
            )
        semantic_requires = tuple(semantic_facts.pop("requires", ()))
        requires = tuple(dict.fromkeys((*implementation.requires, *semantic_requires)))
        combined_facts = {
            **implementation_facts,
            **semantic_facts,
        }
        if requires:
            combined_facts["requires"] = list(requires)
        return Candidate(
            implementation.id,
            {
                "family": implementation.family,
                "variant": implementation.variant,
                **dict(implementation.parameters),
                **semantic,
            },
            combined_facts,
        )

    def choose_default(
        self,
        family: str,
        candidates: tuple[Candidate, ...],
        *,
        portable_fallback: str | None = None,
    ) -> str:
        return self.implementation_model.choose_default(
            family,
            tuple(candidate.id for candidate in candidates),
            portable_fallback=portable_fallback,
        )


class TritonCandidateProvider(Protocol):
    op_names: frozenset[str]

    def propose(
        self,
        node: Node,
        context: TritonCandidateContext,
    ) -> TritonCandidateProposal | None: ...


class TritonCandidateProviderRegistry:
    """One reviewed provider per semantic op, with deterministic registration."""

    def __init__(self) -> None:
        self._providers: list[TritonCandidateProvider] = []
        self._by_op: dict[str, TritonCandidateProvider] = {}

    @property
    def op_names(self) -> frozenset[str]:
        return frozenset(self._by_op)

    @property
    def providers(self) -> tuple[TritonCandidateProvider, ...]:
        return tuple(self._providers)

    def add(self, provider: TritonCandidateProvider) -> None:
        if not provider.op_names:
            raise CodegenError(
                f"Triton candidate provider {type(provider).__name__} owns no ops."
            )
        duplicates = tuple(sorted(provider.op_names & self._by_op.keys()))
        if duplicates:
            owners = {
                op: type(self._by_op[op]).__name__
                for op in duplicates
            }
            raise CodegenError(
                f"Triton candidate provider {type(provider).__name__} duplicates "
                f"reviewed op ownership {owners}."
            )
        self._providers.append(provider)
        for op in provider.op_names:
            self._by_op[op] = provider

    def provider_for(self, op: str) -> TritonCandidateProvider | None:
        return self._by_op.get(op)


__all__ = [
    "TritonCandidateContext",
    "TritonCandidateProposal",
    "TritonCandidateProvider",
    "TritonCandidateProviderRegistry",
]
