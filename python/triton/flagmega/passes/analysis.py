# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit, hash-keyed pass analyses without an IoC container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from triton.flagmega.ir import IRModule


class AnalysisProvider(Protocol):
    name: str

    def analyze(self, module: IRModule, function: str | None = None) -> object: ...


@dataclass(frozen=True)
class AnalysisCacheKey:
    name: str
    semantic_hash: str
    function: str | None


class AnalysisManager:
    """Own lazy analysis providers and results for one PassManager run.

    Results are keyed by semantic hash and optional function scope.  A pass
    that declares an analysis in ``preserves`` explicitly promotes its input
    result to the output hash; all other results are invalidated.
    """

    def __init__(self) -> None:
        self._providers: dict[str, AnalysisProvider] = {}
        self._cache: dict[AnalysisCacheKey, object] = {}
        self._seeded: dict[str, object] = {}

    @property
    def cached_names(self) -> frozenset[str]:
        return frozenset((*self._seeded, *(key.name for key in self._cache)))

    def register(self, provider: AnalysisProvider) -> None:
        name = str(provider.name)
        if not name:
            raise ValueError("Analysis provider requires a non-empty name.")
        if name in self._providers and self._providers[name] is not provider:
            raise ValueError(f"Analysis provider {name!r} is already registered.")
        self._providers[name] = provider

    def seed(self, name: str, value: object = True) -> None:
        if not name:
            raise ValueError("Seeded analysis requires a non-empty name.")
        self._seeded[str(name)] = value

    def require(
        self,
        name: str,
        module: IRModule,
        *,
        function: str | None = None,
    ) -> object:
        name = str(name)
        if name in self._seeded:
            return self._seeded[name]
        key = AnalysisCacheKey(name, module.semantic_hash, function)
        if key in self._cache:
            return self._cache[key]
        try:
            provider = self._providers[name]
        except KeyError as error:
            raise KeyError(
                f"Analysis {name!r} is not registered; available: {sorted(self._providers)}"
            ) from error
        result = provider.analyze(module, function)
        self._cache[key] = result
        return result

    def finish_pass(
        self,
        *,
        input_hash: str,
        output_hash: str,
        preserves: frozenset[str],
    ) -> frozenset[str]:
        preserved = frozenset(str(value) for value in preserves)
        invalidated = self.cached_names - preserved
        if input_hash != output_hash:
            promoted = {
                AnalysisCacheKey(key.name, output_hash, key.function): value
                for key, value in self._cache.items()
                if key.semantic_hash == input_hash and key.name in preserved
            }
            self._cache.update(promoted)
        self._cache = {
            key: value for key, value in self._cache.items()
            if key.name in preserved
        }
        self._seeded = {
            name: value for name, value in self._seeded.items()
            if name in preserved
        }
        return invalidated


__all__ = ["AnalysisCacheKey", "AnalysisManager", "AnalysisProvider"]
