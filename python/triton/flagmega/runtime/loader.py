# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Validated artifact-to-runtime-module loader."""

from __future__ import annotations

import importlib.util
import os
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
    package_registry.register("elementwise_add/v1", "iluvatar-bi-v150", GeneratedAddModule)
    package_registry.register("elementwise/v2", "iluvatar-bi-v150", GeneratedElementwiseModule)
    package_registry.register("tir_call_graph/v1", "iluvatar-bi-v150", create_tir_runtime)


_register_builtin_packages()


def load(path: str | Path, *, device: str | None = None):
    artifact = Path(path).resolve()
    manifest, module = load_artifact(artifact)
    codegen = manifest.get("codegen")
    if manifest.get("status") != "executable" or not isinstance(codegen, Mapping):
        raise ArtifactError("FlagMega artifact does not contain an executable package.")
    kind = str(codegen.get("kind", ""))
    target = str(manifest.get("target", ""))
    if target == "iluvatar-bi-v150":
        # CUDA's default lazy module loading breaks this COREX backend's
        # device-code resolution -- vendor-confirmed 2026-09-20, and
        # independently reproduced this session (perf-iteration/
        # ITERATION.md Trial 49): with lazy loading, a real decode call
        # that normally completes in ~10s instead reliably hung and
        # produced a fresh `dmesg` "XID: 24 mmu page fault" on every one
        # of 3/3 tested GPUs; forcing eager loading here made the same
        # calls pass cleanly on the same GPUs immediately after. A CTA
        # that faults this way never reaches its next in-kernel grid
        # barrier, so every other CTA in that barrier's group spins
        # forever -- this is very likely the actual mechanism behind
        # what otherwise looks like a probabilistic megakernel livelock.
        # Must be set before CUDA initializes in this process (the driver
        # reads it once); setting it here, before this function's first
        # device touch, covers every caller that reaches COREX only
        # through this loader, but cannot help if the calling process
        # already touched a CUDA device beforehand.
        os.environ["CUDA_MODULE_LOADING"] = "0"
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
    kernel = _resolve_kernel(generated, codegen, source)
    result = adapter.factory(artifact, manifest, module, kernel)
    return result.load(device) if device is not None else result


def _resolve_kernel(generated, codegen: Mapping[str, object], source: Path):
    """Resolve a manifest's entry kernel(s) from the generated module.

    Every artifact today has a single top-level `codegen["symbol"]` and
    resolves to one JITFunction, exactly as before. Artifacts that opt in
    to a kernel sequence (kernel-launch boundaries instead of in-kernel
    grid barriers -- see perf-iteration/ITERATION.md Trial 31-32) describe
    it via `codegen["kernels"]`, a list of `{"symbol": ...}` entries in
    launch order; this resolves each and returns a tuple instead of a
    single JITFunction, which `GeneratedTirCallGraphModule.__init__`
    distinguishes via `isinstance(kernel, tuple)`.
    """
    raw_kernels = codegen.get("kernels")
    if raw_kernels is None:
        symbol = str(codegen.get("symbol", ""))
        try:
            return getattr(generated, symbol)
        except AttributeError as error:
            raise ArtifactError(f"Generated source has no entry symbol {symbol!r}.") from error
    symbols = tuple(str(spec["symbol"]) for spec in raw_kernels)
    resolved = []
    for symbol in symbols:
        try:
            resolved.append(getattr(generated, symbol))
        except AttributeError as error:
            raise ArtifactError(
                f"Generated source {source} has no entry symbol {symbol!r} "
                f"(from codegen['kernels'])."
            ) from error
    return tuple(resolved)


__all__ = ["load"]
