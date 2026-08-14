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
"""Load the single local Manifest that indexes remote FlagTune packages.

The Manifest is read from ``FLAGTUNE_LOCAL_MANIFEST`` when configured, otherwise
from ``$FLAGTUNE_MODEL_CACHE/manifest.json``. When the default path is absent and
remote resolution is allowed, the bundled catalog is published there once.
Runtime model loading never fetches or refreshes Manifest metadata. The local
JSON file maps each platform and strict-SemVer version to one HTTPS package URL
and SHA-256 digest; download, package validation, and immutable cache publication
belong to :mod:`model_loader`.

The optional ``latest`` field is descriptive only. When no exact version is
requested, selection computes the highest strict SemVer key in ``versions``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from triton.flagtune.contract.archive import parse_model_version, validate_model_version
from triton.flagtune.contract.identity import validate_platform_key

logger = logging.getLogger(__name__)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class RemotePackage:
    version: str
    url: str
    sha256: str


class ManifestContractError(RuntimeError):
    """Reject a local Manifest that violates schema 1."""


def _reject_duplicate_manifest_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ManifestContractError(f"duplicate JSON key in FlagTune Manifest: {key!r}")
        result[key] = value
    return result


def _manifest_path() -> Path:
    """Return the explicit Manifest path or the model-cache default."""
    if "FLAGTUNE_LOCAL_MANIFEST" in os.environ:
        configured = os.environ["FLAGTUNE_LOCAL_MANIFEST"].strip()
        if not configured:
            raise ManifestContractError("FLAGTUNE_LOCAL_MANIFEST is present but empty")
        return Path(configured).expanduser()
    from triton.flagtune.runtime.model_loader import _cache_root

    return _cache_root() / "manifest.json"


def _manifest_is_valid(data: Any) -> bool:
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int:
        return False
    if set(data) - {"schema_version", "packages"}:
        return False
    if data["schema_version"] != 1:
        return False
    packages = data.get("packages")
    if not isinstance(packages, dict):
        return False
    for platform_key, entry in packages.items():
        try:
            validated_platform_key = validate_platform_key(platform_key, "manifest packages key")
        except ValueError:
            return False
        if validated_platform_key != platform_key:
            return False
        if not isinstance(entry, dict):
            return False
        if set(entry) - {"versions", "latest"}:
            return False
        if "latest" in entry and not isinstance(entry["latest"], str):
            return False
        versions = entry.get("versions")
        if not isinstance(versions, dict):
            return False
        for version, metadata in versions.items():
            try:
                validate_model_version(version)
            except ValueError:
                return False
            if not isinstance(metadata, dict):
                return False
            if set(metadata) != {"url", "sha256"}:
                return False
            location = metadata.get("url")
            digest = metadata.get("sha256")
            if not isinstance(location, str) or not isinstance(digest, str):
                return False
            parsed = urlparse(location.strip())
            if parsed.scheme.lower() != "https" or not parsed.netloc:
                return False
            if _SHA256_RE.fullmatch(digest) is None:
                return False
    return True


def _load_manifest(*, generate_default: bool = True) -> Dict[str, Any]:
    """Read and validate the single deployment-controlled Manifest."""
    path = _manifest_path()
    if path.is_symlink():
        raise ManifestContractError(f"FlagTune Manifest must not be a symlink: {path}")
    if generate_default and "FLAGTUNE_LOCAL_MANIFEST" not in os.environ and not path.exists():
        try:
            from triton.flagtune.training.manifest_generator import ensure_default_manifest

            created = ensure_default_manifest(path)
            if created:
                logger.info("Generated default FlagTune Manifest at %s", path)
        except (ImportError, OSError, ValueError) as exc:
            raise ManifestContractError(f"cannot generate default FlagTune Manifest {path}: {exc}") from exc
    if path.is_symlink():
        raise ManifestContractError(f"FlagTune Manifest must not be a symlink: {path}")
    if not path.is_file():
        raise ManifestContractError(f"FlagTune Manifest is not a regular file: {path}")
    try:
        resolved = path.resolve(strict=True)
        with resolved.open("r", encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=_reject_duplicate_manifest_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestContractError(f"cannot read valid JSON from FlagTune Manifest {path}: {exc}") from exc
    if not _manifest_is_valid(data):
        raise ManifestContractError(f"FlagTune Manifest {path} does not satisfy schema 1")
    return data


def _selected_version(entry: Any, requested: Optional[str]) -> Optional[str]:
    if not isinstance(entry, dict):
        return None
    versions = entry.get("versions")
    if not isinstance(versions, dict):
        return None
    if requested is not None:
        try:
            return validate_model_version(requested)
        except ValueError:
            return None
    candidates = []
    for candidate in versions:
        try:
            parsed = parse_model_version(candidate)
        except ValueError:
            continue
        candidates.append((parsed.selection_key, candidate))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def resolve_package_info(
    platform_key: str,
    *,
    version: Optional[str] = None,
    generate_default: bool = True,
) -> Optional[RemotePackage]:
    """Resolve one package URL and digest from the active local Manifest."""
    platform_key = validate_platform_key(platform_key)
    manifest = _load_manifest(generate_default=generate_default)
    entry = manifest["packages"].get(platform_key)
    if not isinstance(entry, dict):
        return None
    versions = entry.get("versions")
    selected = _selected_version(entry, version)
    if not isinstance(versions, dict) or selected is None:
        return None
    metadata = versions.get(selected)
    if not isinstance(metadata, dict):
        return None
    url = metadata.get("url")
    digest = metadata.get("sha256")
    if not isinstance(url, str) or not isinstance(digest, str):
        return None
    url = url.strip()
    parsed_url = urlparse(url)
    if parsed_url.scheme.lower() != "https" or not parsed_url.netloc:
        return None
    if _SHA256_RE.fullmatch(digest) is None:
        return None
    return RemotePackage(selected, url, digest)
