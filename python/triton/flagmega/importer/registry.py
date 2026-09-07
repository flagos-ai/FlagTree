# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit architecture/model-type importer registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from triton.flagmega.errors import ImporterError
from triton.flagmega.importer.checkpoint import Checkpoint
from triton.flagmega.ir import IRModule


LayerImporter = Callable[[Checkpoint, int, str | None], IRModule]
ModelImporter = Callable[[Checkpoint, str | None], IRModule]


@dataclass(frozen=True)
class ModelImporterSpec:
    name: str
    architectures: frozenset[str]
    model_types: frozenset[str]
    import_layer: LayerImporter
    import_model: ModelImporter | None = None
    numerical_profiles: Mapping[str, Callable[[IRModule], IRModule]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "architectures", frozenset(str(value) for value in self.architectures))
        object.__setattr__(self, "model_types", frozenset(str(value) for value in self.model_types))
        if not self.name or not self.architectures and not self.model_types:
            raise ImporterError("Importer spec requires a name and architecture or model_type key.")
        if not callable(self.import_layer) or self.import_model is not None and not callable(self.import_model):
            raise ImporterError(f"Importer spec {self.name!r} callbacks must be callable.")
        profiles = dict(self.numerical_profiles)
        if any(not isinstance(name, str) or not name or name == "nncase" or not callable(callback)
               for name, callback in profiles.items()):
            raise ImporterError("Numerical profiles require nonempty names and callable import transforms; nncase is reserved.")
        object.__setattr__(self, "numerical_profiles", MappingProxyType(profiles))

    def numerical_transform(self, profile: str) -> Callable[[IRModule], IRModule]:
        if profile == "nncase":
            return lambda module: module
        if profile not in self.numerical_profiles:
            raise ImporterError(
                f"Importer {self.name!r} does not support numerical profile {profile!r}; "
                f"available: {('nncase', *self.numerical_profiles)}.")
        return self.numerical_profiles[profile]

    def matches(self, config: Mapping[str, object]) -> bool:
        architectures = {str(value) for value in config.get("architectures", ())}
        model_type = str(config.get("model_type", ""))
        return bool(self.architectures & architectures or model_type in self.model_types)


class ModelImporterRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ModelImporterSpec] = {}

    def register(self, spec: ModelImporterSpec) -> ModelImporterSpec:
        if spec.name in self._specs:
            raise ImporterError(f"Importer spec {spec.name!r} is already registered.")
        self._specs[spec.name] = spec
        return spec

    def resolve(self, config: Mapping[str, object], *, full_model: bool) -> ModelImporterSpec:
        matched = tuple(spec for spec in self.specs if spec.matches(config))
        architectures = tuple(str(value) for value in config.get("architectures", ()))
        model_type = str(config.get("model_type", ""))
        if not matched:
            raise ImporterError(
                f"FlagMega has no {'full-model' if full_model else 'layer'} importer for "
                f"architectures={architectures!r}, model_type={model_type!r}.")
        if len(matched) != 1:
            raise ImporterError(
                f"Importer dispatch is ambiguous for architectures={architectures!r}, "
                f"model_type={model_type!r}: {[value.name for value in matched]}.")
        spec = matched[0]
        if full_model and spec.import_model is None:
            raise ImporterError(
                f"Importer {spec.name!r} supports layer import but has no full-model importer.")
        return spec

    @property
    def specs(self) -> tuple[ModelImporterSpec, ...]:
        return tuple(self._specs[name] for name in sorted(self._specs))


__all__ = [
    "LayerImporter",
    "ModelImporter",
    "ModelImporterRegistry",
    "ModelImporterSpec",
]
