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
"""Load the local or remote Manifest that indexes FlagTune packages.

The Manifest is read from ``FLAGTUNE_LOCAL_MANIFEST`` when configured. Otherwise
the cached Manifest is refreshed from ``FLAGTUNE_MANIFEST_URL`` when its TTL
expires. A failed remote refresh is reported to the caller instead of silently
using stale metadata. The remote URL must point to a tar.gz containing exactly
``manifest.json``. Its schema is validated before cache publication.

The optional ``latest`` field is descriptive only. When no exact version is
requested, selection computes the highest strict SemVer key in ``versions``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tarfile
import tempfile
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request

from triton.flagtune.contract.archive import parse_model_version, validate_model_version
from triton.flagtune.contract.identity import validate_platform_key

logger = logging.getLogger(__name__)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MANIFEST_MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
_MANIFEST_MAX_MEMBER_BYTES = 4 * 1024 * 1024
_DEFAULT_MANIFEST_TTL = 24 * 60 * 60


@dataclass(frozen=True)
class RemotePackage:
    version: str
    url: str
    sha256: str


class ManifestContractError(RuntimeError):
    """Reject a Manifest that violates schema 1."""


class ManifestFetchError(ManifestContractError):
    """Report a transient or transport failure while fetching a Manifest."""


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


def _manifest_meta_path(path: Path) -> Path:
    return path.with_name("manifest.meta.json")


def _manifest_url() -> str:
    return os.environ.get("FLAGTUNE_MANIFEST_URL", "").strip()


def _environment_switch(name: str) -> bool:
    value = os.environ.get(name, "").strip()
    if value in ("", "0"):
        return False
    if value == "1":
        return True
    raise ManifestContractError(f"{name} must be 0 or 1, got {value!r}")


def _remote_disabled() -> bool:
    return _environment_switch("FLAGTUNE_DISABLE_REMOTE")


def _manifest_ttl() -> int:
    value = os.environ.get("FLAGTUNE_MANIFEST_TTL", "").strip()
    if not value:
        return _DEFAULT_MANIFEST_TTL
    try:
        ttl = int(value)
    except ValueError as exc:
        raise ManifestContractError(f"FLAGTUNE_MANIFEST_TTL must be a non-negative integer, got {value!r}") from exc
    if ttl < 0:
        raise ManifestContractError(f"FLAGTUNE_MANIFEST_TTL must be a non-negative integer, got {value!r}")
    return ttl


def _manifest_cache_is_fresh(path: Path) -> bool:
    if _environment_switch("FLAGTUNE_MANIFEST_REFRESH"):
        return False
    metadata_path = _manifest_meta_path(path)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fetched_at = metadata.get("fetched_at")
        if metadata.get("source") != "remote" or not isinstance(fetched_at, (int, float)):
            return False
        _read_manifest_file(path)
        return time.time() - float(fetched_at) < _manifest_ttl()
    except (
            OSError,
            UnicodeError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            ManifestContractError,
    ):
        return False


def _read_manifest_bytes(payload: bytes, source: str) -> Dict[str, Any]:
    try:
        data = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_manifest_keys)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestContractError(f"cannot read valid JSON from FlagTune Manifest {source}: {exc}") from exc
    if not _manifest_is_valid(data):
        raise ManifestContractError(f"FlagTune Manifest {source} does not satisfy schema 1")
    return data


def _read_manifest_file(path: Path) -> Dict[str, Any]:
    if path.is_symlink():
        raise ManifestContractError(f"FlagTune Manifest must not be a symlink: {path}")
    if not path.is_file():
        raise ManifestContractError(f"FlagTune Manifest is not a regular file: {path}")
    try:
        payload = path.read_bytes()
    except (OSError, UnicodeError) as exc:
        raise ManifestContractError(f"cannot read FlagTune Manifest {path}: {exc}") from exc
    return _read_manifest_bytes(payload, str(path))


def _atomic_write(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise ManifestContractError(f"FlagTune Manifest cache must not be a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _bundle_members(payload: bytes, source: str) -> bytes:
    if len(payload) > _MANIFEST_MAX_ARCHIVE_BYTES:
        raise ManifestFetchError(f"remote Manifest archive is too large: {len(payload)} bytes")
    files: Dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
            for member in archive.getmembers():
                if member.name != "manifest.json":
                    raise ManifestContractError(f"remote Manifest archive contains unexpected member {member.name!r}")
                if (member.name in files or not member.isfile() or member.isdir() or member.issym() or member.islnk()):
                    raise ManifestContractError(f"remote Manifest archive contains invalid member {member.name!r}")
                if member.size < 0 or member.size > _MANIFEST_MAX_MEMBER_BYTES:
                    raise ManifestFetchError(f"remote Manifest member {member.name!r} is too large")
                handle = archive.extractfile(member)
                if handle is None:
                    raise ManifestContractError(f"remote Manifest member {member.name!r} is not readable")
                content = handle.read(_MANIFEST_MAX_MEMBER_BYTES + 1)
                if len(content) > _MANIFEST_MAX_MEMBER_BYTES:
                    raise ManifestFetchError(f"remote Manifest member {member.name!r} is too large")
                files[member.name] = content
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise ManifestFetchError(f"cannot read remote Manifest archive {source}: {exc}") from exc
    if set(files) != {"manifest.json"}:
        raise ManifestFetchError("remote Manifest archive must contain exactly manifest.json")
    return files["manifest.json"]


def _fetch_remote_manifest() -> tuple[Dict[str, Any], bytes, str]:
    url = _manifest_url()
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise ManifestFetchError(f"FLAGTUNE_MANIFEST_URL must be an HTTPS URL, got {url!r}")
    try:
        from triton.flagtune.runtime.model_loader import _open_https

        request = Request(url, headers={"User-Agent": "FlagTune/manifest"})
        with _open_https(request, timeout=20) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > _MANIFEST_MAX_ARCHIVE_BYTES:
                raise ManifestFetchError("remote Manifest archive is too large")
            chunks = []
            size = 0
            while True:
                chunk = response.read(min(1024 * 1024, _MANIFEST_MAX_ARCHIVE_BYTES - size + 1))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > _MANIFEST_MAX_ARCHIVE_BYTES:
                    raise ManifestFetchError("remote Manifest archive is too large")
    except ManifestFetchError:
        raise
    except (HTTPError, URLError, OSError, ValueError, TypeError) as exc:
        raise ManifestFetchError(f"cannot fetch remote FlagTune Manifest {url}: {exc}") from exc
    manifest_bytes = _bundle_members(b"".join(chunks), url)
    return _read_manifest_bytes(manifest_bytes, url), manifest_bytes, url


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


def _load_manifest() -> Dict[str, Any]:
    """Read an explicit local, fresh cached, or remote Manifest."""
    path = _manifest_path()
    explicit = "FLAGTUNE_LOCAL_MANIFEST" in os.environ
    if path.is_symlink():
        raise ManifestContractError(f"FlagTune Manifest must not be a symlink: {path}")
    if explicit:
        return _read_manifest_file(path)

    remote_url = _manifest_url()
    remote_disabled = _remote_disabled()
    if path.is_file():
        if not remote_url or remote_disabled or _manifest_cache_is_fresh(path):
            return _read_manifest_file(path)
    elif path.exists():
        raise ManifestContractError(f"FlagTune Manifest is not a regular file: {path}")
    if remote_disabled:
        raise ManifestFetchError(
            f"FlagTune Manifest is not cached at {path} and FLAGTUNE_DISABLE_REMOTE=1 prevents downloading it")
    if not remote_url:
        raise ManifestFetchError(
            f"FLAGTUNE_MANIFEST_URL is not configured and no cached FlagTune Manifest exists at {path}")

    data, manifest_bytes, source_url = _fetch_remote_manifest()
    _atomic_write(path, manifest_bytes)
    metadata = {
        "source": "remote",
        "source_url": source_url,
        "fetched_at": time.time(),
    }
    _atomic_write(
        _manifest_meta_path(path),
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    logger.info("Updated cached FlagTune Manifest from %s", source_url)
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
) -> Optional[RemotePackage]:
    """Resolve one package URL and digest from the active Manifest."""
    platform_key = validate_platform_key(platform_key)
    manifest = _load_manifest()
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
    configured_base = os.environ.get("FLAGTUNE_MODEL_BASE_URL", "").strip()
    if configured_base:
        mirror = urlparse(configured_base)
        filename = parsed_url.path.rsplit("/", 1)[-1]
        if mirror.scheme.lower() != "https" or not mirror.netloc or not filename:
            raise ManifestContractError("FLAGTUNE_MODEL_BASE_URL must be an HTTPS URL with a model path")
        url = f"{configured_base.rstrip('/')}/{filename}"
    return RemotePackage(selected, url, digest)
