# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Validated artifact-to-runtime-module loader."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Mapping

from triton.flagmega.artifacts import load_artifact
from triton.flagmega.artifacts.manifest import resolve_artifact_path
from triton.flagmega.errors import ArtifactError
from triton.flagmega.runtime.module import (
    GeneratedAddModule,
    GeneratedElementwiseModule,
    create_tir_runtime,
)
from triton.flagmega.runtime.registry import package_registry


def _register_builtin_packages() -> None:
    package_registry.register("elementwise_add/v1", "nvidia-sm90", GeneratedAddModule)
    package_registry.register("elementwise/v2", "nvidia-sm90", GeneratedElementwiseModule)
    package_registry.register("tir_call_graph/v1", "nvidia-sm90", create_tir_runtime)


_register_builtin_packages()


def load(path: str | Path, *, device: str | None = None):
    artifact = Path(path).resolve()
    manifest, module = load_artifact(artifact)
    codegen = manifest.get("codegen")
    if manifest.get("status") != "executable" or not isinstance(codegen, Mapping):
        raise ArtifactError("FlagMega artifact does not contain an executable package.")
    kind = str(codegen.get("kind", ""))
    target = str(manifest.get("target", ""))
    adapter = package_registry.resolve(kind, target)
    source = resolve_artifact_path(artifact, str(codegen.get("source", "")))
    module_name = f"_flagmega_artifact_{manifest['semantic_hash'][:16]}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ArtifactError(f"Cannot import generated kernel source {source}.")
    generated = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(generated)
    except Exception as error:
        raise ArtifactError(f"Cannot import generated kernel source {source}: {error}.") from error
    symbol = str(codegen.get("symbol", ""))
    try:
        kernel = getattr(generated, symbol)
    except AttributeError as error:
        raise ArtifactError(f"Generated source has no entry symbol {symbol!r}.") from error
    result = adapter.factory(artifact, manifest, module, kernel)
    return result.load(device) if device is not None else result


__all__ = ["load"]
