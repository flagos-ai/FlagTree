# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Versioned, content-addressed sections in a FlagMega artifact."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

from triton.flagmega.errors import ArtifactError
from triton.flagmega.ir import IRModule


ARTIFACT_SCHEMA = "flagmega.artifact/v2"
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class ArtifactSection:
    """One immutable file referenced by the artifact manifest."""

    name: str
    kind: str
    path: str
    nbytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.name or not self.kind:
            raise ArtifactError("Artifact section name and kind must be non-empty.")
        if self.nbytes < 0:
            raise ArtifactError(f"Artifact section {self.name!r} has a negative size.")
        if len(self.sha256) != 64 or any(value not in "0123456789abcdef" for value in self.sha256):
            raise ArtifactError(f"Artifact section {self.name!r} has an invalid SHA-256 digest.")
        _validate_relative_path(self.path)

    def to_data(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "path": self.path,
            "nbytes": self.nbytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_data(cls, data: object) -> ArtifactSection:
        if not isinstance(data, Mapping):
            raise ArtifactError("Artifact section must be an object.")
        try:
            return cls(
                name=str(data["name"]),
                kind=str(data["kind"]),
                path=str(data["path"]),
                nbytes=int(data["nbytes"]),
                sha256=str(data["sha256"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactError(f"Malformed artifact section: {error}.") from error


def hash_file(path: str | Path) -> str:
    """Return a SHA-256 digest without materializing the complete file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def record_section(
    root: str | Path,
    path: str | Path,
    *,
    name: str,
    kind: str,
    known_sha256: str | None = None,
    known_nbytes: int | None = None,
) -> ArtifactSection:
    """Record one immutable artifact section.

    Producers that computed a digest while creating a large file may pass both
    ``known_sha256`` and ``known_nbytes``.  The size is still checked against
    the file system; the digest is trusted only as producer-owned metadata and
    will be verified when the artifact is loaded.
    """

    artifact = Path(root).resolve()
    source = Path(path).resolve()
    try:
        relative = source.relative_to(artifact)
    except ValueError as error:
        raise ArtifactError(f"Artifact section {source} is outside {artifact}.") from error
    if not source.is_file():
        raise ArtifactError(f"Artifact section {name!r} is missing: {source}.")
    if (known_sha256 is None) != (known_nbytes is None):
        raise ArtifactError("Known artifact section digest and size must be provided together.")
    actual_nbytes = source.stat().st_size
    if known_nbytes is not None:
        if actual_nbytes != known_nbytes:
            raise ArtifactError(
                f"Artifact section {name!r} has {actual_nbytes} bytes, "
                f"but its producer reported {known_nbytes}."
            )
        digest = known_sha256
    else:
        digest = hash_file(source)
    assert digest is not None
    return ArtifactSection(name, kind, relative.as_posix(), actual_nbytes, digest)


def parse_sections(data: object) -> tuple[ArtifactSection, ...]:
    if not isinstance(data, list):
        raise ArtifactError("Artifact manifest sections must be a list.")
    sections = tuple(ArtifactSection.from_data(value) for value in data)
    names = [section.name for section in sections]
    paths = [section.path for section in sections]
    if len(set(names)) != len(names):
        raise ArtifactError("Artifact section names must be unique.")
    if len(set(paths)) != len(paths):
        raise ArtifactError("Artifact section paths must be unique.")
    return sections


def verify_sections(
    root: str | Path,
    sections: tuple[ArtifactSection, ...],
    *,
    defer_hashes: frozenset[str] = frozenset(),
) -> None:
    """Verify manifest sections, optionally delegating named hashes downstream.

    Deferred sections still receive path, existence, and size validation.  A
    caller must subsequently validate their bytes against the same digest and
    cross-check any subordinate index.  This is used for rdata because its
    verifier hashes the image and all indexed tensor ranges in one pass.
    """

    for section in sections:
        path = resolve_artifact_path(root, section.path)
        if not path.is_file():
            raise ArtifactError(f"Artifact section {section.name!r} is missing: {path}.")
        if path.stat().st_size != section.nbytes:
            raise ArtifactError(f"Artifact section {section.name!r} size does not match its manifest.")
        if section.name not in defer_hashes and hash_file(path) != section.sha256:
            raise ArtifactError(f"Artifact section {section.name!r} hash does not match its manifest.")


def section_map(sections: tuple[ArtifactSection, ...]) -> dict[str, ArtifactSection]:
    return {section.name: section for section in sections}


def require_sections(
    sections: Mapping[str, ArtifactSection],
    required: Mapping[str, str],
) -> None:
    for name, kind in required.items():
        try:
            section = sections[name]
        except KeyError as error:
            raise ArtifactError(f"Artifact is missing required section {name!r}.") from error
        if section.kind != kind:
            raise ArtifactError(
                f"Artifact section {name!r} has kind {section.kind!r}, expected {kind!r}.")


def resolve_artifact_path(root: str | Path, relative: str) -> Path:
    _validate_relative_path(relative)
    artifact = Path(root).resolve()
    path = (artifact / relative).resolve()
    try:
        path.relative_to(artifact)
    except ValueError as error:
        raise ArtifactError(f"Artifact path {relative!r} escapes {artifact}.") from error
    return path


def module_abi(module: IRModule) -> dict[str, object]:
    """Return the runtime-visible function/type contract, independent of codegen."""

    nodes = module.node_map
    functions = []
    for function in module.functions:
        functions.append({
            "name": function.name,
            "parameters": [
                {
                    "node": node_id,
                    "name": str(nodes[node_id].attrs.get("name", node_id)),
                    "type": nodes[node_id].type.to_data(),
                }
                for node_id in function.parameters
            ],
            "outputs": [
                {"node": node_id, "type": nodes[node_id].type.to_data()}
                for node_id in function.outputs
            ],
        })
    return {"entry": module.entry, "functions": functions}


def write_json_atomic(path: str | Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def _validate_relative_path(value: str) -> None:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ArtifactError(f"Artifact section path must be a safe relative path, got {value!r}.")


__all__ = [
    "ARTIFACT_SCHEMA",
    "ArtifactSection",
    "hash_file",
    "module_abi",
    "parse_sections",
    "record_section",
    "require_sections",
    "resolve_artifact_path",
    "section_map",
    "verify_sections",
    "write_json_atomic",
]
