# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit entry-package renderer registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import IRModule


PackageMatcher = Callable[[IRModule], bool]
DescriptorBuilder = Callable[[IRModule], dict[str, object]]
SourceRenderer = Callable[[dict[str, object], str], str]


@dataclass(frozen=True)
class PackageRendererSpec:
    name: str
    kind: str
    matches: PackageMatcher
    describe: DescriptorBuilder
    render: SourceRenderer

    def __post_init__(self) -> None:
        if not self.name or not self.kind or "/" not in self.kind:
            raise CodegenError("Package renderer requires a name and versioned kind.")
        if not all(callable(value) for value in (self.matches, self.describe, self.render)):
            raise CodegenError(f"Package renderer {self.name!r} callbacks must be callable.")


class PackageRendererRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, PackageRendererSpec] = {}

    def register(self, spec: PackageRendererSpec) -> PackageRendererSpec:
        if spec.name in self._specs:
            raise CodegenError(f"Package renderer {spec.name!r} is already registered.")
        if any(value.kind == spec.kind for value in self._specs.values()):
            raise CodegenError(f"Package kind {spec.kind!r} is already registered.")
        self._specs[spec.name] = spec
        return spec

    def resolve(self, module: IRModule) -> PackageRendererSpec:
        matched = tuple(spec for spec in self.specs if spec.matches(module))
        if not matched:
            raise CodegenError(
                f"No Triton entry-package renderer matches stage={module.stage!r}, "
                f"entry={module.entry!r}.",
                stage=module.stage,
            )
        if len(matched) != 1:
            raise CodegenError(
                f"Triton entry-package renderer match is ambiguous: {[value.name for value in matched]}.",
                stage=module.stage,
            )
        return matched[0]

    @property
    def specs(self) -> tuple[PackageRendererSpec, ...]:
        return tuple(self._specs[name] for name in sorted(self._specs))


__all__ = [
    "DescriptorBuilder",
    "PackageMatcher",
    "PackageRendererRegistry",
    "PackageRendererSpec",
    "SourceRenderer",
]
