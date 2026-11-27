# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Target-neutral registry for selecting implementations of semantic TIR ops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from triton.flagmega.codegen.triton.implementation import (
    TritonImplementation,
    TritonImplementationModel,
)
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import Candidate, IRModule, KernelDispatch, PrimFunction


@dataclass(frozen=True)
class TIRMicroKernelProposal:
    """Editable implementation alternatives for one semantic PrimFunction."""

    candidates: tuple[Candidate, ...]
    default_candidate: str

    def __post_init__(self) -> None:
        candidate_ids = tuple(candidate.id for candidate in self.candidates)
        if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
            raise CodegenError(
                "A TIR microkernel proposal requires non-empty unique candidates."
            )
        if self.default_candidate not in candidate_ids:
            raise CodegenError(
                f"TIR microkernel default {self.default_candidate!r} is not in "
                f"the proposal {candidate_ids}."
            )


@dataclass(frozen=True)
class TIRMicroKernelContext:
    """Semantic TIR and target catalog visible to a portable provider."""

    module: IRModule
    function: PrimFunction
    dispatch: KernelDispatch
    implementation_model: TritonImplementationModel

    def implementations(
        self, family: str, **contract: object
    ) -> tuple[TritonImplementation, ...]:
        return self.implementation_model.find(family, **contract)

    def candidate(self, implementation: TritonImplementation) -> Candidate:
        facts = dict(implementation.facts)
        if implementation.requires:
            facts["requires"] = implementation.requires
        return Candidate(
            implementation.id,
            dict(implementation.parameters),
            facts,
        )

    def choose_default(
        self, family: str, candidates: tuple[Candidate, ...]
    ) -> str:
        return self.implementation_model.choose_default(
            family, tuple(candidate.id for candidate in candidates)
        )


class TIRMicroKernelProvider(Protocol):
    op_names: frozenset[str]

    def propose(
        self, context: TIRMicroKernelContext
    ) -> TIRMicroKernelProposal | None: ...


class TIRMicroKernelProviderRegistry:
    """Deterministic semantic-op ownership independent of target vendors."""

    def __init__(self) -> None:
        self._providers: list[TIRMicroKernelProvider] = []
        self._by_op: dict[str, TIRMicroKernelProvider] = {}

    @property
    def op_names(self) -> frozenset[str]:
        return frozenset(self._by_op)

    @property
    def providers(self) -> tuple[TIRMicroKernelProvider, ...]:
        return tuple(self._providers)

    def add(self, provider: TIRMicroKernelProvider) -> None:
        if not provider.op_names:
            raise CodegenError(
                f"TIR microkernel provider {type(provider).__name__} owns no ops."
            )
        duplicates = tuple(sorted(provider.op_names & self._by_op.keys()))
        if duplicates:
            raise CodegenError(
                f"TIR microkernel provider {type(provider).__name__} duplicates "
                f"semantic op ownership {duplicates}."
            )
        self._providers.append(provider)
        for op_name in provider.op_names:
            self._by_op[op_name] = provider

    def provider_for(self, op_name: str) -> TIRMicroKernelProvider | None:
        return self._by_op.get(op_name)


__all__ = [
    "TIRMicroKernelContext",
    "TIRMicroKernelProposal",
    "TIRMicroKernelProvider",
    "TIRMicroKernelProviderRegistry",
]
