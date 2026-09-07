# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Stable FlagMega compiler artifact with a strict section/ABI manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping
from typing import TYPE_CHECKING

from triton.flagmega.errors import ArtifactError
from triton.flagmega.importer.checkpoint import Checkpoint
from triton.flagmega.ir import IRModule, companion_suffix, emit_module, load_module, text_source, verify_module
from triton.flagmega.artifacts.rdata import pack_rdata, verify_rdata
from triton.flagmega.artifacts.manifest import (
    ARTIFACT_SCHEMA,
    module_abi,
    parse_sections,
    record_section,
    require_sections,
    resolve_artifact_path,
    section_map,
    verify_sections,
    write_json_atomic,
)
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.codegen.triton.diagnostics import dump_schedules

if TYPE_CHECKING:
    from triton.flagmega.diagnostics import Dumper


ARTIFACT_VERSION = 2


def write_artifact(
    module: IRModule,
    output_dir: str | Path,
    *,
    target: str,
    checkpoint: Checkpoint | None = None,
    emit_executable: bool = False,
    dumper: Dumper | None = None,
    rdata_cache_dir: str | Path | None = None,
) -> Path:
    verify_module(module)
    plan = None
    if module.stage == "bufferized_tir":
        from triton.flagmega.ir import verify_buffer_plan

        plan = verify_buffer_plan(module)
    if checkpoint is not None and module.stage != "bufferized_tir":
        raise ArtifactError("Readonly data can only be packed from bufferized_tir.")
    if emit_executable and plan is not None and plan.rdata_bytes and checkpoint is None:
        raise ArtifactError(
            "Executable artifacts with readonly-data buffers require a checkpoint to materialize rdata; "
            "pass --checkpoint when compiling or resuming edited Python IR.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    ir_dir = destination / "ir"
    ir_dir.mkdir(exist_ok=True)
    final_ir = emit_module(module, ir_dir / "final.py")
    rdata = None
    if checkpoint is not None:
        rdata = pack_rdata(
            module,
            checkpoint,
            destination / "assets",
            cache_dir=rdata_cache_dir,
        )
    codegen = (
        render_triton_package(module, destination, dumper=dumper)
        if emit_executable
        else None
    )
    if plan is not None and not emit_executable:
        dump_schedules(module, dumper)
    sections = [
        record_section(destination, final_ir, name="ir.python", kind="ir.python/v1"),
        record_section(
            destination,
            final_ir.with_suffix(companion_suffix(module)),
            name="ir.text",
            kind="ir.text/v1",
        ),
    ]
    if rdata is not None:
        sections.extend((
            record_section(
                destination,
                destination / "assets" / "rdata.index.json",
                name="rdata.index",
                kind="rdata.index/v1",
            ),
            record_section(
                destination,
                destination / "assets" / "rdata.bin",
                name="rdata.image",
                kind="rdata.image/v1",
                known_sha256=str(rdata["sha256"]),
                known_nbytes=int(rdata["nbytes"]),
            ),
        ))
    if codegen is not None:
        sections.append(record_section(
            destination,
            destination / str(codegen["source"]),
            name="codegen.source",
            kind="codegen.triton-python/v1",
        ))
    manifest = {
        "flagmega_artifact_version": ARTIFACT_VERSION,
        "schema": ARTIFACT_SCHEMA,
        "status": "executable" if codegen is not None else "compiler_ir",
        "target": target,
        "entry": module.entry,
        "dialect": module.dialect,
        "stage": module.stage,
        "semantic_hash": module.semantic_hash,
        "abi": module_abi(module),
        "sections": [section.to_data() for section in sections],
        "final_ir": str(final_ir.relative_to(destination)),
        "final_text": str(final_ir.with_suffix(companion_suffix(module)).relative_to(destination)),
        "selections": [record.to_data() for record in module.selections],
        "metadata": module.to_data()["metadata"],
        "rdata": None if rdata is None else {
            "index": "assets/rdata.index.json",
            "image": "assets/rdata.bin",
            "nbytes": rdata["nbytes"],
            "sha256": rdata["sha256"],
        },
        "codegen": codegen,
    }
    write_json_atomic(destination / "artifact.json", manifest)
    return destination


def load_artifact(path: str | Path) -> tuple[dict[str, object], IRModule]:
    artifact = Path(path).resolve()
    manifest_path = artifact / "artifact.json"
    if not manifest_path.is_file():
        raise ArtifactError(f"FlagMega artifact manifest is missing: {manifest_path}.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"Cannot read FlagMega artifact manifest {manifest_path}: {error}.") from error
    if not isinstance(manifest, dict):
        raise ArtifactError("FlagMega artifact manifest must be a JSON object.")
    version = manifest.get("flagmega_artifact_version")
    if version == 1:
        return _load_legacy_v1(artifact, manifest)
    if version != ARTIFACT_VERSION or manifest.get("schema") != ARTIFACT_SCHEMA:
        raise ArtifactError(
            f"Unsupported FlagMega artifact version/schema {version!r}/{manifest.get('schema')!r}.")
    sections = parse_sections(manifest.get("sections"))
    rdata_descriptor = manifest.get("rdata")
    deferred_hashes = (
        frozenset({"rdata.image"})
        if rdata_descriptor is not None
        else frozenset()
    )
    verify_sections(artifact, sections, defer_hashes=deferred_hashes)
    by_name = section_map(sections)
    required = {"ir.python": "ir.python/v1", "ir.text": "ir.text/v1"}
    if manifest.get("rdata") is not None:
        required.update({"rdata.index": "rdata.index/v1", "rdata.image": "rdata.image/v1"})
    if manifest.get("status") == "executable":
        required["codegen.source"] = "codegen.triton-python/v1"
    elif manifest.get("status") != "compiler_ir":
        raise ArtifactError(f"Unsupported artifact status {manifest.get('status')!r}.")
    require_sections(by_name, required)
    if str(manifest.get("final_ir")) != by_name["ir.python"].path:
        raise ArtifactError("Artifact final_ir does not identify the ir.python section.")
    if str(manifest.get("final_text")) != by_name["ir.text"].path:
        raise ArtifactError("Artifact final_text does not identify the ir.text section.")
    module = _load_ir(resolve_artifact_path(artifact, by_name["ir.python"].path))
    _verify_manifest_module_contract(manifest, module)
    final_text = resolve_artifact_path(artifact, by_name["ir.text"].path)
    if final_text.read_text(encoding="utf-8") != text_source(module):
        raise ArtifactError("FlagMega artifact compact IL/script dump is stale.")
    rdata = rdata_descriptor
    if rdata is not None:
        if not isinstance(rdata, Mapping):
            raise ArtifactError("Artifact rdata descriptor must be an object or null.")
        if str(rdata.get("index")) != by_name["rdata.index"].path:
            raise ArtifactError("Artifact rdata index path does not match its section.")
        if str(rdata.get("image")) != by_name["rdata.image"].path:
            raise ArtifactError("Artifact rdata image path does not match its section.")
        index = verify_rdata(resolve_artifact_path(artifact, by_name["rdata.index"].path).parent, module=module)
        if index["sha256"] != rdata.get("sha256") or index["nbytes"] != rdata.get("nbytes"):
            raise ArtifactError("FlagMega artifact rdata summary does not match its verified index.")
        image_section = by_name["rdata.image"]
        if index["sha256"] != image_section.sha256 or index["nbytes"] != image_section.nbytes:
            raise ArtifactError("FlagMega artifact rdata image section does not match its verified index.")
    codegen = manifest.get("codegen")
    if manifest.get("status") == "executable":
        if not isinstance(codegen, Mapping):
            raise ArtifactError("Executable artifact must contain a codegen descriptor.")
        source = by_name["codegen.source"]
        if str(codegen.get("source")) != source.path or codegen.get("source_sha256") != source.sha256:
            raise ArtifactError("Artifact codegen descriptor does not match its source section.")
        if not str(codegen.get("kind", "")) or not str(codegen.get("symbol", "")):
            raise ArtifactError("Artifact codegen descriptor requires non-empty kind and symbol.")
    elif codegen is not None:
        raise ArtifactError("compiler_ir artifact cannot contain a codegen descriptor.")
    return manifest, module


def _load_ir(path: Path) -> IRModule:
    try:
        return load_module(path)
    except ArtifactError:
        raise
    except Exception as error:
        raise ArtifactError(f"Cannot load artifact IR {path}: {error}.") from error


def _verify_manifest_module_contract(manifest: Mapping[str, object], module: IRModule) -> None:
    if module.semantic_hash != manifest.get("semantic_hash"):
        raise ArtifactError("FlagMega artifact semantic hash does not match final IR.")
    for field, actual in (
        ("entry", module.entry),
        ("dialect", module.dialect),
        ("stage", module.stage),
    ):
        if manifest.get(field) != actual:
            raise ArtifactError(f"Artifact {field} does not match final IR.")
    target = manifest.get("target")
    if not isinstance(target, str) or not target:
        raise ArtifactError("Artifact target must be a non-empty string.")
    if manifest.get("abi") != module_abi(module):
        raise ArtifactError("Artifact runtime ABI does not match final IR functions and types.")


def _load_legacy_v1(artifact: Path, manifest: dict[str, object]) -> tuple[dict[str, object], IRModule]:
    """Read existing v1 build artifacts without weakening the v2 writer contract."""

    final_ir = resolve_artifact_path(artifact, str(manifest.get("final_ir", "")))
    module = _load_ir(final_ir)
    if module.semantic_hash != manifest.get("semantic_hash"):
        raise ArtifactError("FlagMega artifact semantic hash does not match final IR.")
    final_text = resolve_artifact_path(artifact, str(manifest.get("final_text", "")))
    if not final_text.is_file() or final_text.read_text(encoding="utf-8") != text_source(module):
        raise ArtifactError("FlagMega artifact compact IL/script dump is missing or stale.")
    rdata = manifest.get("rdata")
    if rdata is not None:
        if not isinstance(rdata, Mapping):
            raise ArtifactError("Legacy artifact rdata descriptor must be an object.")
        index_path = resolve_artifact_path(artifact, str(rdata.get("index", "")))
        index = verify_rdata(index_path.parent, module=module)
        if index["sha256"] != rdata.get("sha256") or index["nbytes"] != rdata.get("nbytes"):
            raise ArtifactError("FlagMega artifact rdata summary does not match its verified index.")
    codegen = manifest.get("codegen")
    if codegen is not None:
        if not isinstance(codegen, Mapping):
            raise ArtifactError("Legacy artifact codegen descriptor must be an object.")
        source = resolve_artifact_path(artifact, str(codegen.get("source", "")))
        if not source.is_file():
            raise ArtifactError("Legacy artifact generated source is missing.")
        import hashlib

        if hashlib.sha256(source.read_bytes()).hexdigest() != codegen.get("source_sha256"):
            raise ArtifactError("Legacy artifact generated source does not match its hash.")
    return manifest, module
