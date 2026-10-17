# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Stable FlagMega compiler artifact with a strict section/ABI manifest."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

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
from triton.flagmega.artifacts.rdata import RDATA_COMPOSITE_DIGEST_KINDS, pack_rdata, verify_rdata
from triton.flagmega.artifacts.source_layout import SOURCE_LAYOUT_DIGEST
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.codegen.triton.diagnostics import dump_schedules
from triton.flagmega.errors import ArtifactError, IRSchemaError
from triton.flagmega.importer.checkpoint import Checkpoint
from triton.flagmega.ir import (
    IRModule,
    KernelDefinition,
    companion_suffix,
    emit_module,
    load_module,
    text_source,
    verify_module,
)
from triton.flagmega.ir.model import _to_data

if TYPE_CHECKING:
    from triton.flagmega.diagnostics import Dumper


ARTIFACT_VERSION = 2

# Codegen backends an executable artifact can be emitted through. Both consume
# the same selected, bufferized TIR; they differ only in what they emit and how
# the runtime launches it.
CODEGEN_BACKENDS = ("triton", "cpp")

# Section kind per backend, so a manifest states which one produced it and
# `load_artifact` can require the right sections.
_CODEGEN_SECTION_KINDS = {
    "triton": "codegen.triton-python/v1",
    "cpp": "codegen.cpp-source/v1",
}


def _image_kind(digest_kind):
    if digest_kind == SOURCE_LAYOUT_DIGEST:
        return "rdata.image/v3"
    return "rdata.image/v2" if digest_kind in RDATA_COMPOSITE_DIGEST_KINDS else "rdata.image/v1"


def write_artifact(
    module: IRModule,
    output_dir: str | Path,
    *,
    target: str,
    checkpoint: Checkpoint | None = None,
    emit_executable: bool = False,
    codegen_backend: str = "triton",
    dumper: Dumper | None = None,
    rdata_cache_dir: str | Path | None = None,
) -> Path:
    verify_module(module)
    if codegen_backend not in CODEGEN_BACKENDS:
        raise ArtifactError(
            f"Unknown codegen backend {codegen_backend!r}; "
            f"expected one of {list(CODEGEN_BACKENDS)}."
        )
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
    # IR and source emission read immutable compiler objects; packing writes
    # disjoint asset files and spends its time in NumPy, hashing and I/O. Start
    # both producers together rather than putting serialization on the weight
    # materialization critical path. Publish the manifest only after both join.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="flagmega-assets") as worker:
        rdata_future = (
            worker.submit(pack_rdata, module, checkpoint, destination / "assets", cache_dir=rdata_cache_dir)
            if checkpoint is not None else None
        )
        final_ir = emit_module(module, ir_dir / "final.py")
        codegen = (
            _render_codegen_package(module, destination, target=target, codegen_backend=codegen_backend, dumper=dumper)
            if emit_executable else None
        )
        rdata = rdata_future.result() if rdata_future is not None else None
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
                kind=_image_kind(rdata.get("digest_kind")),
                known_sha256=str(rdata["sha256"]),
                known_nbytes=int(rdata["nbytes"]),
            ),
        ))
        for index, source in enumerate(rdata.get("sources", [])):
            sections.append(record_section(
                destination, destination / "assets" / source["path"],
                name=f"rdata.source.{index}", kind="rdata.source/v1",
                known_sha256=source["sha256"], known_nbytes=source["nbytes"],
            ))
    if codegen is not None:
        sections.append(record_section(
            destination,
            destination / str(codegen["source"]),
            name="codegen.source",
            kind=_CODEGEN_SECTION_KINDS[codegen_backend],
        ))
        # A C++ package is only loadable through its built library, so the
        # library is a verified section too: a hand-edited or stale .so is
        # rejected at load time exactly as a stale generated source is.
        if codegen_backend == "cpp":
            sections.append(record_section(
                destination,
                destination / str(codegen["library"]),
                name="codegen.library",
                kind="codegen.cpp-library/v1",
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
        "metadata": _to_data(module.metadata),
        "rdata": None if rdata is None else {
            "index": "assets/rdata.index.json",
            "image": "assets/rdata.bin",
            "nbytes": rdata["nbytes"],
            "sha256": rdata["sha256"],
            "digest_kind": rdata.get("digest_kind", "sha256"),
        },
        "codegen": codegen,
        # Stated even for a compiler_ir artifact: "which backend would emit
        # this" is a property of the compile, and the loader needs it before
        # it has looked at the codegen descriptor.
        "codegen_backend": codegen_backend,
    }
    write_json_atomic(destination / "artifact.json", manifest)
    return destination


def _render_codegen_package(
    module: IRModule,
    destination: Path,
    *,
    target: str,
    codegen_backend: str,
    dumper: Dumper | None,
) -> dict[str, object]:
    if codegen_backend == "triton":
        return render_triton_package(module, destination, dumper=dumper)
    from triton.flagmega.codegen.cpp import render_cpp_package_artifact

    return render_cpp_package_artifact(
        module, destination, target=target, dumper=dumper
    )


def refresh_artifact_codegen(
    path: str | Path, *, dumper: Dumper | None = None,
    kernel_rewrite: Callable[[IRModule], Sequence[KernelDefinition]] | None = None,
) -> Path:
    """Re-render kernels while retaining the artifact's closed readonly snapshot.

    Readonly bytes and their integrity descriptors are retained verbatim.
    Payload checking remains at the runtime mapping boundary, as for ordinary
    metadata loads. An optional rewrite may change kernel dispatch attributes
    and schedules; graph, constants, buffer signatures, effects and workspaces
    stay fixed. Such an edit publishes a new verified IR checkpoint as well.
    This entry point accepts no replacement checkpoint or readonly data.
    The caller owns publication to this directory; readers can keep using the
    old source while a new content-addressed source is staged and the manifest
    is replaced atomically.
    """
    artifact = Path(path).resolve()
    manifest_path = artifact / "artifact.json"
    before = manifest_path.read_bytes() if manifest_path.is_file() else None
    manifest, module = load_artifact(artifact, verify_rdata_payload=False)
    if (manifest.get("status") != "executable"
            or manifest.get("codegen_backend", "triton") != "triton"
            or manifest.get("flagmega_artifact_version") != ARTIFACT_VERSION):
        raise ArtifactError("Codegen refresh requires a current executable Triton artifact.")
    original_hash = module.semantic_hash
    if kernel_rewrite is not None:
        try:
            definitions = tuple(kernel_rewrite(module))
            originals = {value.name: value for value in module.kernel_definitions}
            if (len(definitions) != len(originals) or {value.name for value in definitions} != set(originals)):
                raise ArtifactError("Kernel rewrite must preserve the complete kernel ABI.")
            for value in definitions:
                old = originals[value.name]
                expected = replace(old, dispatch=replace(old.dispatch, microkernel=value.dispatch.microkernel,
                                                         semantic_attrs=value.dispatch.semantic_attrs))
                if (value != expected
                        or value.dispatch.microkernel.shared_workspaces != old.dispatch.microkernel.shared_workspaces
                        or value.dispatch.microkernel.transfer_pipeline != old.dispatch.microkernel.transfer_pipeline):
                    raise ArtifactError("Kernel rewrite must preserve kernel signatures, effects and workspace ABI.")
            module = verify_module(replace(module, kernel_definitions=definitions))
        except IRSchemaError as error:
            raise ArtifactError(f"Kernel rewrite violates the verified kernel signature/ABI: {error}") from error
    with tempfile.TemporaryDirectory(prefix=".flagmega-codegen-", dir=artifact) as staging:
        staging = Path(staging)
        replacement_sections = {}
        if module.semantic_hash != original_hash:
            final_ir = emit_module(module, staging / f"final-{module.semantic_hash}.py")
            for file, name, kind in ((final_ir, "ir.python", "ir.python/v1"),
                                     (final_ir.with_suffix(companion_suffix(module)), "ir.text", "ir.text/v1")):
                destination = resolve_artifact_path(artifact, f"ir/{file.name}")
                os.replace(file, destination)
                replacement_sections[name] = record_section(artifact, destination, name=name, kind=kind).to_data()
            manifest = {**manifest, "semantic_hash": module.semantic_hash,
                        "final_ir": replacement_sections["ir.python"]["path"],
                        "final_text": replacement_sections["ir.text"]["path"]}
        codegen = render_triton_package(module, staging, dumper=dumper)
        source = resolve_artifact_path(staging, str(codegen["source"]))
        relative = f"codegen/{codegen['source_sha256']}.py"
        destination = resolve_artifact_path(artifact, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
        replacement_sections["codegen.source"] = record_section(
            artifact, destination, name="codegen.source", kind=_CODEGEN_SECTION_KINDS["triton"]).to_data()
        codegen = {**codegen, "source": relative}
        sections = [replacement_sections.get(value["name"], value) for value in manifest["sections"]]
        if manifest_path.read_bytes() != before:
            raise ArtifactError("Artifact manifest changed during codegen refresh; refusing to overwrite it.")
        write_json_atomic(manifest_path, {**manifest, "codegen": codegen, "sections": sections})
    return artifact


def load_artifact(
    path: str | Path,
    *,
    verify_rdata_payload: bool = True,
) -> tuple[dict[str, object], IRModule]:
    """Load and verify an artifact's manifest, sections and IR.

    ``verify_rdata_payload=False`` defers only the rdata payload digest to the
    caller that maps those bytes, which is where the chip segment to read is
    known.  Every metadata and contract check still runs.  Callers whose purpose
    is to audit the artifact itself must leave it enabled.
    """

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
        return _load_legacy_v1(
            artifact, manifest, verify_rdata_payload=verify_rdata_payload
        )
    if version != ARTIFACT_VERSION or manifest.get("schema") != ARTIFACT_SCHEMA:
        raise ArtifactError(
            f"Unsupported FlagMega artifact version/schema {version!r}/{manifest.get('schema')!r}.")
    sections = parse_sections(manifest.get("sections"))
    rdata_descriptor = manifest.get("rdata")
    deferred_hashes = (
        frozenset({"rdata.image", *(section.name for section in sections if section.kind == "rdata.source/v1")})
        if rdata_descriptor is not None
        else frozenset()
    )
    verify_sections(artifact, sections, defer_hashes=deferred_hashes)
    by_name = section_map(sections)
    required = {"ir.python": "ir.python/v1", "ir.text": "ir.text/v1"}
    if rdata_descriptor is not None:
        if not isinstance(rdata_descriptor, Mapping):
            raise ArtifactError("Artifact rdata descriptor must be an object or null.")
        image_kind = _image_kind(rdata_descriptor.get("digest_kind"))
        required.update({"rdata.index": "rdata.index/v1", "rdata.image": image_kind})
    # Older v2 artifacts predate the backend field; they are all Triton.
    codegen_backend = str(manifest.get("codegen_backend", "triton"))
    if codegen_backend not in CODEGEN_BACKENDS:
        raise ArtifactError(
            f"Artifact declares unknown codegen backend {codegen_backend!r}."
        )
    if manifest.get("status") == "executable":
        required["codegen.source"] = _CODEGEN_SECTION_KINDS[codegen_backend]
        if codegen_backend == "cpp":
            required["codegen.library"] = "codegen.cpp-library/v1"
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
        # The payload digest is deferred to the component that maps the bytes, the
        # same delegation ``defer_hashes`` already applies to the image section.
        # Loading inspects metadata; it never reads the payload, and for a
        # chip-sharded image it cannot know which segment this rank will map.
        # Hashing here would cost a full pass over the image that the runtime
        # then repeats, scoped correctly, moments later.
        index = verify_rdata(
            resolve_artifact_path(artifact, by_name["rdata.index"].path).parent,
            module=module,
            verify_image_bytes=verify_rdata_payload,
        )
        if index["sha256"] != rdata.get("sha256") or index["nbytes"] != rdata.get("nbytes"):
            raise ArtifactError("FlagMega artifact rdata summary does not match its verified index.")
        if index.get("digest_kind", "sha256") != rdata.get("digest_kind", "sha256"):
            raise ArtifactError("FlagMega artifact rdata digest kind does not match its verified index.")
        image_section = by_name["rdata.image"]
        expected_kind = _image_kind(index.get("digest_kind"))
        if image_section.kind != expected_kind:
            raise ArtifactError("FlagMega artifact rdata image section digest kind does not match its index.")
        if index["sha256"] != image_section.sha256 or index["nbytes"] != image_section.nbytes:
            raise ArtifactError("FlagMega artifact rdata image section does not match its verified index.")
        source_sections = {name: section for name, section in by_name.items() if section.kind == "rdata.source/v1"}
        expected_sources = {f"rdata.source.{i}": source for i, source in enumerate(index.get("sources", []))}
        if set(source_sections) != set(expected_sources):
            raise ArtifactError("FlagMega artifact source sections do not cover its rdata snapshots.")
        for name, source in expected_sources.items():
            section = source_sections[name]
            if (section.path != f"assets/{source['path']}" or section.nbytes != source["nbytes"]
                    or section.sha256 != source["sha256"]):
                raise ArtifactError("FlagMega artifact source section differs from its rdata index.")
    codegen = manifest.get("codegen")
    if manifest.get("status") == "executable":
        if not isinstance(codegen, Mapping):
            raise ArtifactError("Executable artifact must contain a codegen descriptor.")
        source = by_name["codegen.source"]
        if str(codegen.get("source")) != source.path or codegen.get("source_sha256") != source.sha256:
            raise ArtifactError("Artifact codegen descriptor does not match its source section.")
        if not str(codegen.get("kind", "")) or not str(codegen.get("symbol", "")):
            raise ArtifactError("Artifact codegen descriptor requires non-empty kind and symbol.")
        if codegen_backend == "cpp":
            library = by_name["codegen.library"]
            if (
                str(codegen.get("library")) != library.path
                or codegen.get("library_sha256") != library.sha256
            ):
                raise ArtifactError(
                    "Artifact codegen descriptor does not match its built "
                    "library section."
                )
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


def _load_legacy_v1(
    artifact: Path,
    manifest: dict[str, object],
    *,
    verify_rdata_payload: bool = True,
) -> tuple[dict[str, object], IRModule]:
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
        index = verify_rdata(
            index_path.parent, module=module, verify_image_bytes=verify_rdata_payload
        )
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
