# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Validated artifact-to-runtime-module loader."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Mapping
from types import SimpleNamespace

from triton.flagmega.artifacts import load_artifact
from triton.flagmega.artifacts.manifest import resolve_artifact_path
from triton.flagmega.artifacts.program_sources import program_source_records
from triton.flagmega.errors import ArtifactError
from triton.flagmega.runtime.cpp_module import create_cpp_tir_runtime
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
    package_registry.register("elementwise_add/v1", "nvidia-sm120", GeneratedAddModule)
    package_registry.register("elementwise/v2", "nvidia-sm120", GeneratedElementwiseModule)
    package_registry.register("tir_call_graph/v1", "nvidia-sm120", create_tir_runtime)
    package_registry.register("elementwise_add/v1", "iluvatar-bi-v150", GeneratedAddModule)
    package_registry.register("elementwise/v2", "iluvatar-bi-v150", GeneratedElementwiseModule)
    package_registry.register("tir_call_graph/v1", "iluvatar-bi-v150", create_tir_runtime)
    package_registry.register("tir_call_graph_cpp/v1", "iluvatar-bi-v150", create_cpp_tir_runtime)
    for target in ("nvidia-sm89", "nvidia-sm90", "nvidia-sm120"):
        package_registry.register("tir_call_graph_cpp/v1", target, create_cpp_tir_runtime)
    # Ada (sm_89, e.g. RTX 4090) runs the same portable TIR call graph.
    package_registry.register("tir_call_graph/v1", "nvidia-sm89", create_tir_runtime)
    # The TIR call-graph runtime is model- and platform-neutral; the RDNA path
    # differs only in device validation (HIP vs CUDA capability).
    package_registry.register("tir_call_graph/v1", "amd-gfx1100", create_tir_runtime)
    package_registry.register("tir_call_graph/v1", "amd-gfx1201", create_tir_runtime)


_register_builtin_packages()


def load(path: str | Path, *, device: str | None = None, shared_rdata=None):
    artifact = Path(path).resolve()
    # The runtime adapter digests the rdata payload when it maps the pool, scoped
    # to the chip segment this rank reads, so loading must not hash it as well.
    manifest, module = load_artifact(artifact, verify_rdata_payload=False)
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
    if str(manifest.get("codegen_backend", "triton")) == "cpp":
        # A C++ package is already compiled; its generated source is kept for
        # inspection and hash verification, not imported. The adapter opens the
        # artifact's built library itself, so there is no kernel object here.
        kernel = None
    else:
        source = resolve_artifact_path(artifact, str(codegen.get("source", "")))
        module_name = f"_flagmega_artifact_{manifest['semantic_hash'][:16]}"
        records = program_source_records(codegen)
        if records:
            generated = _load_program_sources(artifact, codegen, records, module_name)
        else:
            generated = _import_source(source, module_name)
        kernel = _resolve_kernel(generated, codegen, source)
    result = adapter.factory(artifact, manifest, module, kernel)
    if shared_rdata is not None:
        from triton.flagmega.runtime.module import GeneratedTirCallGraphModule
        if device is None or not isinstance(result, GeneratedTirCallGraphModule):
            raise ArtifactError("Shared readonly loading requires a device and a TIR call graph.")
        return result.load(device, shared_rdata=shared_rdata)
    return result.load(device) if device is not None else result


def _import_source(source, module_name):
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ArtifactError(f"Cannot import generated kernel source {source}.")
    generated = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(generated)
    except Exception as error:
        raise ArtifactError(f"Cannot import generated kernel source {source}: {error}.") from error
    return generated


def _load_program_sources(artifact, codegen, records, module_name):
    entries = {}
    for record in records:
        source = resolve_artifact_path(artifact, record["source"])
        generated = _import_source(source, f"{module_name}_{record['symbol']}")
        for symbol in record["symbols"]:
            try:
                entries[symbol] = getattr(generated, symbol)
            except AttributeError as error:
                raise ArtifactError(f"Program source {source} has no entry {symbol!r}.") from error
    aliases = codegen.get("program_entry_aliases", {})
    for symbol in aliases:
        target = symbol
        while target in aliases:
            target = aliases[target]
        entries[symbol] = entries[target]
    return SimpleNamespace(**entries)


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
