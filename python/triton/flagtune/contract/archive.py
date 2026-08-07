# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Create, validate, and order self-contained FlagTune model archives.

An exported model is one deterministic ``model.tar.gz`` rather than an
unpacked directory.  Its required root-level members are the XGBoost ranker,
the compiled YAML contract, and the training summary.  :mod:`training` writes
these archives and :mod:`model_manager` reads them before loading a predictor.

This module deliberately validates an archive in memory and never calls
``TarFile.extract*``.  That prevents path traversal, links, duplicate members,
and other archive-layout surprises, but it also means every member is held in
memory and archives may contain only root-level regular files.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Mapping, Tuple

from triton.flagtune.contract.identity import (
    ModelIdentity,
    validate_platform_key,
)

MODEL_ARCHIVE_NAME = "model.tar.gz"
PACKAGE_MANIFEST_NAME = "package_manifest.json"
REQUIRED_MODEL_MEMBERS = (
    "xgboost_ranker.json",
    "flagtune_config.yaml",
    "training_summary.json",
)

# Strict Semantic Versioning 2.0 grammar used for artifact directory names.
# Examples accepted: ``1.2.3``, ``1.2.3-rc.1``, and ``1.2.3+build.7``.
# Examples rejected: ``v1.2.3``, ``1.2``, and ``01.2.3``.  Build metadata is
# retained as text for deterministic selection, but does not affect SemVer
# precedence; therefore ``1.2.3+cpu`` and ``1.2.3+gpu`` have equal precedence.
_SEMVER_RE = re.compile(r"^(0|[1-9][0-9]*)\."
                        r"(0|[1-9][0-9]*)\."
                        r"(0|[1-9][0-9]*)"
                        r"(?:-((?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
                        r"(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*))?"
                        r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$")


class ModelArchiveError(ValueError):
    """Report an unsafe, incomplete, or malformed model archive."""


@dataclass(frozen=True)
class PlatformPackage:
    """Hold a validated platform Manifest and its child archive payloads."""

    platform_key: str
    package_version: str
    models: Mapping[str, Mapping[str, str]]
    archives: Mapping[str, bytes]


@dataclass(frozen=True)
class SemanticVersion:
    """A strict SemVer 2.0 value with precedence-aware ordering data."""

    text: str
    major: int
    minor: int
    patch: int
    prerelease: Tuple[str, ...]

    @property
    def precedence_key(self) -> tuple:
        """Return a key implementing SemVer precedence (excluding build data)."""
        identifiers = tuple((0, int(value)) if value.isdigit() else (1, value) for value in self.prerelease)
        # A release has higher precedence than every prerelease of that release.
        return (self.major, self.minor, self.patch, not self.prerelease, identifiers)

    @property
    def selection_key(self) -> tuple:
        """Add the complete text as a deterministic tie-break for build metadata."""
        return (*self.precedence_key, self.text)


def parse_model_version(value: str) -> SemanticVersion:
    """Parse a strict SemVer 2.0 model revision without normalizing its text.

    Args:
        value: Candidate archive revision, such as ``"1.4.0-rc.2"``.

    Returns:
        Parsed numeric components and prerelease identifiers for selection.

    Raises:
        ValueError: If ``value`` is not strict SemVer 2.0.
    """
    if not isinstance(value, str) or not _SEMVER_RE.fullmatch(value):
        raise ValueError(f"model version must be strict SemVer 2.0: {value!r}")
    match = _SEMVER_RE.fullmatch(value)
    assert match is not None
    prerelease = tuple(match.group(4).split(".")) if match.group(4) else ()
    return SemanticVersion(value, int(match.group(1)), int(match.group(2)), int(match.group(3)), prerelease)


def validate_model_version(value: str) -> str:
    """Validate and return a model version suitable for one path segment."""
    return parse_model_version(value).text


def platform_package_name(platform_key: str, version: str) -> str:
    """Return the canonical filename for one versioned platform package."""
    return f"{validate_platform_key(platform_key)}_v{validate_model_version(version)}.tar.gz"


def _platform_model_path(identity: ModelIdentity) -> str:
    return f"{identity.op_id}/{identity.variant}/{identity.dtype_key}/{MODEL_ARCHIVE_NAME}"


def _validate_platform_member(member: tarfile.TarInfo, seen: set[str]) -> str:
    name = member.name
    relative = PurePosixPath(name)
    if (not name or name in (".", "..") or relative.is_absolute() or name != relative.as_posix() or "\\" in name
            or any(part in (".", "..") for part in relative.parts)):
        raise ModelArchiveError(f"platform package member must be a safe relative path: {name!r}")
    if name in seen:
        raise ModelArchiveError(f"duplicate platform package member: {name!r}")
    if not member.isfile():
        raise ModelArchiveError(f"platform package member is not a regular file: {name!r}")
    seen.add(name)
    return name


def _validate_child_archive(payload: bytes, identity: ModelIdentity, version: str, source: str) -> None:
    members = read_model_archive_bytes(payload, source=source)
    try:
        from triton.flagtune.contract.operator_schema import load_model_config_bytes, model_identity_from_config

        _, config = load_model_config_bytes(members["flagtune_config.yaml"], source=source)
        declared_identity = model_identity_from_config(config)
    except (TypeError, ValueError) as exc:
        raise ModelArchiveError(f"invalid child model config at {source}: {exc}") from exc
    if declared_identity != identity:
        raise ModelArchiveError(f"child model identity mismatch at {source}: "
                                f"{declared_identity.artifact_key!r} != {identity.artifact_key!r}")
    if config.get("model_version") != version:
        raise ModelArchiveError(
            f"child model version mismatch at {source}: {config.get('model_version')!r} != {version!r}")


def _write_deterministic_tar(path: Path, members: Mapping[str, bytes]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                    for name, payload in members.items():
                        info = tarfile.TarInfo(name)
                        info.size = len(payload)
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mode = 0o644
                        archive.addfile(info, io.BytesIO(payload))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return path


def write_platform_package(
        path: Path | str,
        *,
        platform_key: str,
        package_version: str,
        model_archives: Mapping[ModelIdentity, Path | str],
        required_identities=(),
) -> Path:
    """Write a reproducible platform package containing indexed child archives."""
    platform = validate_platform_key(platform_key)
    version = validate_model_version(package_version)
    missing = sorted(set(required_identities) - set(model_archives), key=lambda identity: identity.artifact_key)
    if missing:
        missing_keys = [item.artifact_key for item in missing]
        raise ModelArchiveError(f"platform package is missing required identities: {missing_keys}")
    models: Dict[str, Dict[str, str]] = {}
    archives: Dict[str, bytes] = {}
    for identity, archive_path in sorted(model_archives.items(), key=lambda item: item[0].artifact_key):
        if not isinstance(identity, ModelIdentity):
            raise TypeError("platform package model archive keys must be ModelIdentity values")
        if identity.platform_key != platform:
            raise ModelArchiveError(
                f"model identity platform does not match package: {identity.platform_key!r} != {platform!r}")
        member_path = _platform_model_path(identity)
        payload = Path(archive_path).read_bytes()
        _validate_child_archive(payload, identity, version, str(archive_path))
        models[identity.artifact_key] = {"path": member_path}
        archives[member_path] = payload
    manifest = {
        "models": models,
        "package_version": version,
        "platform_key": platform,
        "schema_version": 1,
    }
    manifest_payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    members = {PACKAGE_MANIFEST_NAME: manifest_payload}
    members.update((name, archives[name]) for name in sorted(archives))
    return _write_deterministic_tar(Path(path), members)


def _identity_from_artifact_key(value: str) -> ModelIdentity:
    parts = value.split("/") if isinstance(value, str) else []
    if len(parts) < 5:
        raise ModelArchiveError(f"invalid model artifact key: {value!r}")
    try:
        identity = ModelIdentity(parts[0], "/".join(parts[1:-2]), parts[-2], parts[-1])
    except ValueError as exc:
        raise ModelArchiveError(f"invalid model artifact key {value!r}: {exc}") from exc
    if value != identity.artifact_key:
        raise ModelArchiveError(f"model artifact key must be canonical: {value!r}")
    return identity


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ModelArchiveError(f"duplicate platform package Manifest key: {key!r}")
        result[key] = value
    return result


def read_platform_package_bytes(
    payload: bytes,
    *,
    expected_platform_key: str,
    expected_version: str,
    source: str,
) -> PlatformPackage:
    """Read one in-memory platform package and return its indexed children."""
    members: Dict[str, bytes] = {}
    seen: set[str] = set()
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            for member in archive.getmembers():
                name = _validate_platform_member(member, seen)
                stream = archive.extractfile(member)
                if stream is None:
                    raise ModelArchiveError(f"cannot read platform package member: {name!r}")
                with stream:
                    members[name] = stream.read()
    except ModelArchiveError:
        raise
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise ModelArchiveError(f"invalid gzip tar platform package at {source}: {exc}") from exc
    if PACKAGE_MANIFEST_NAME not in members:
        raise ModelArchiveError(f"platform package at {source} is missing {PACKAGE_MANIFEST_NAME}")
    try:
        manifest = json.loads(
            members[PACKAGE_MANIFEST_NAME].decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except ModelArchiveError:
        raise
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelArchiveError(f"invalid platform package Manifest at {source}: {exc}") from exc
    root_keys = {"schema_version", "platform_key", "package_version", "models"}
    if not isinstance(manifest, dict) or set(manifest) != root_keys:
        raise ModelArchiveError(f"platform package Manifest must have exact root keys: {sorted(root_keys)}")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ModelArchiveError(f"unsupported platform package schema at {source}")
    platform = manifest["platform_key"]
    version = manifest["package_version"]
    models = manifest["models"]
    try:
        canonical_platform = validate_platform_key(platform, "Manifest platform_key")
        validate_model_version(version)
    except ValueError as exc:
        raise ModelArchiveError(f"invalid platform package Manifest at {source}: {exc}") from exc
    if platform != canonical_platform:
        raise ModelArchiveError(f"platform package Manifest platform_key must be canonical: {platform!r}")
    if not isinstance(models, dict):
        raise ModelArchiveError("platform package Manifest models must be an object")
    if platform != expected_platform_key:
        raise ModelArchiveError(f"platform package platform mismatch: {platform!r} != {expected_platform_key!r}")
    if version != expected_version:
        raise ModelArchiveError(f"platform package version mismatch: {version!r} != {expected_version!r}")

    identities: Dict[str, ModelIdentity] = {}
    indexed_paths: Dict[str, str] = {}
    for artifact, entry in models.items():
        identity = _identity_from_artifact_key(artifact)
        if not isinstance(entry, dict) or set(entry) != {"path"}:
            raise ModelArchiveError(f"platform package model {artifact!r} must have exact entry keys: ['path']")
        path = entry["path"]
        if not isinstance(path, str):
            raise ModelArchiveError(f"platform package model path must be a string for {artifact!r}")
        if path in indexed_paths:
            raise ModelArchiveError(f"duplicate model path {path!r} for {indexed_paths[path]!r} and {artifact!r}")
        indexed_paths[path] = artifact
        identities[artifact] = identity
        if path != _platform_model_path(identity):
            raise ModelArchiveError(f"platform package model path mismatch for {artifact!r}")
        if identity.platform_key != platform:
            raise ModelArchiveError(f"platform package model platform mismatch for {artifact!r}")

    actual_children = set(members) - {PACKAGE_MANIFEST_NAME}
    expected_children = set(indexed_paths)
    missing = sorted(expected_children - actual_children)
    if missing:
        raise ModelArchiveError(f"platform package at {source} is missing indexed children: {missing}")
    unindexed = sorted(actual_children - expected_children)
    if unindexed:
        raise ModelArchiveError(f"platform package at {source} has unindexed children: {unindexed}")

    archives: Dict[str, bytes] = {}
    for path, artifact in indexed_paths.items():
        child_source = f"{source}:{path}"
        _validate_child_archive(members[path], identities[artifact], version, child_source)
        archives[artifact] = members[path]
    return PlatformPackage(platform, version, models, archives)


def read_platform_package(
    path: Path | str,
    *,
    expected_platform_key: str,
    expected_version: str,
) -> PlatformPackage:
    """Read and validate one canonically named on-disk platform package."""
    package_path = Path(path)
    expected_name = platform_package_name(expected_platform_key, expected_version)
    if package_path.name != expected_name:
        raise ModelArchiveError(f"platform package filename mismatch: {package_path.name!r} != {expected_name!r}")
    return read_platform_package_bytes(
        package_path.read_bytes(),
        expected_platform_key=expected_platform_key,
        expected_version=expected_version,
        source=str(package_path),
    )


def _validate_member(member: tarfile.TarInfo, seen: set[str]) -> str:
    name = member.name
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or len(relative.parts) != 1 or name != relative.name
            or relative.parts[0] in (".", "..") or "\\" in name):
        raise ModelArchiveError(f"model archive member must be a root-level file: {name!r}")
    if name in seen:
        raise ModelArchiveError(f"duplicate model archive member: {name!r}")
    if not member.isfile():
        raise ModelArchiveError(f"model archive member is not a regular file: {name!r}")
    seen.add(name)
    return name


def read_model_archive_bytes(payload: bytes, *, source: str = "model archive") -> Dict[str, bytes]:
    """Validate a gzip tar payload and return every safe root member in memory.

    Args:
        payload: Complete gzip-compressed tar payload.
        source: Diagnostic label included in validation errors.

    Returns:
        A mapping from archive member name to its exact bytes.

    Raises:
        ModelArchiveError: If decompression fails, a member is unsafe, or a
            required member is missing.
    """
    members: Dict[str, bytes] = {}
    seen: set[str] = set()
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            for member in archive.getmembers():
                name = _validate_member(member, seen)
                stream = archive.extractfile(member)
                if stream is None:
                    raise ModelArchiveError(f"cannot read model archive member: {name!r}")
                with stream:
                    members[name] = stream.read()
    except ModelArchiveError:
        raise
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise ModelArchiveError(f"invalid gzip tar model archive at {source}: {exc}") from exc
    missing = [name for name in REQUIRED_MODEL_MEMBERS if name not in members]
    if missing:
        raise ModelArchiveError(f"model archive at {source} is missing required members: {missing}")
    return members


def read_model_archive(path: Path | str) -> Dict[str, bytes]:
    """Read and validate one on-disk ``model.tar.gz`` without extracting it."""
    archive_path = Path(path)
    try:
        payload = archive_path.read_bytes()
    except OSError:
        raise
    return read_model_archive_bytes(payload, source=str(archive_path))


def write_model_archive(path: Path | str, members: Mapping[str, bytes]) -> Path:
    """Atomically write a reproducible gzip tar containing root-level files.

    Required entries must be supplied as bytes.  Member order, timestamps,
    ownership, permissions, and gzip metadata are normalized so identical
    inputs produce identical bytes.  The destination is replaced atomically on
    the same filesystem; concurrent writers still race at the final replace.
    """
    target = Path(path)
    missing = [name for name in REQUIRED_MODEL_MEMBERS if name not in members]
    if missing:
        raise ModelArchiveError(f"cannot write model archive without required members: {missing}")
    names = list(REQUIRED_MODEL_MEMBERS) + sorted(set(members) - set(REQUIRED_MODEL_MEMBERS))
    for name in names:
        fake = tarfile.TarInfo(name)
        fake.type = tarfile.REGTYPE
        _validate_member(fake, set())
        if not isinstance(members[name], bytes):
            raise TypeError(f"model archive member {name!r} must be bytes")

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                    for name in names:
                        payload = members[name]
                        info = tarfile.TarInfo(name)
                        info.size = len(payload)
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mode = 0o644
                        archive.addfile(info, io.BytesIO(payload))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return target
