# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit generated-package adapter registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from triton.flagmega.errors import ArtifactError


RuntimeFactory = Callable[[Path, dict[str, object], object, object], object]


@dataclass(frozen=True)
class RuntimePackageSpec:
    kind: str
    target: str
    factory: RuntimeFactory

    def __post_init__(self) -> None:
        if not self.kind or "/" not in self.kind:
            raise ArtifactError(f"Runtime package kind must be versioned, got {self.kind!r}.")
        if not self.target:
            raise ArtifactError("Runtime package target must be non-empty.")
        if not callable(self.factory):
            raise ArtifactError("Runtime package factory must be callable.")


class RuntimePackageRegistry:
    """Maps a versioned codegen package and target to one runtime adapter."""

    def __init__(self) -> None:
        self._specs: dict[tuple[str, str], RuntimePackageSpec] = {}

    def register(self, kind: str, target: str, factory: RuntimeFactory) -> RuntimePackageSpec:
        spec = RuntimePackageSpec(str(kind), str(target), factory)
        key = (spec.kind, spec.target)
        if key in self._specs:
            raise ArtifactError(
                f"Runtime package adapter {spec.kind!r} for {spec.target!r} is already registered.")
        self._specs[key] = spec
        return spec

    def resolve(self, kind: str, target: str) -> RuntimePackageSpec:
        key = (str(kind), str(target))
        try:
            return self._specs[key]
        except KeyError as error:
            available = [
                {"kind": spec.kind, "target": spec.target}
                for spec in sorted(self._specs.values(), key=lambda value: (value.target, value.kind))
            ]
            raise ArtifactError(
                f"Runtime has no adapter for package {kind!r} on target {target!r}; "
                f"available adapters: {available}.") from error

    def create(
        self,
        kind: str,
        target: str,
        artifact: Path,
        manifest: dict[str, object],
        ir_module: object,
        kernel: object,
    ) -> object:
        return self.resolve(kind, target).factory(artifact, manifest, ir_module, kernel)

    @property
    def specs(self) -> tuple[RuntimePackageSpec, ...]:
        return tuple(sorted(self._specs.values(), key=lambda value: (value.target, value.kind)))


package_registry = RuntimePackageRegistry()


__all__ = [
    "RuntimeFactory",
    "RuntimePackageRegistry",
    "RuntimePackageSpec",
    "package_registry",
]
