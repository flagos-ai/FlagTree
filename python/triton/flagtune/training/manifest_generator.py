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
"""Generate a deployment-controlled local FlagTune download Manifest.

Edit the variables in the catalog section to publish more platforms or
versions. A version may use ``filename`` to join against ``MODEL_BASE_URL`` or
an explicit ``url`` to select another HTTPS endpoint.

Run from a source checkout with::

    PYTHONPATH=python python -m triton.flagtune.training.manifest_generator

The output path is ``FLAGTUNE_LOCAL_MANIFEST`` when configured, otherwise
``$FLAGTUNE_MODEL_CACHE/manifest.json``. Neither timestamps nor a ``latest``
field are emitted; runtime selection computes the highest strict SemVer key.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from triton.flagtune.contract.archive import validate_model_version
from triton.flagtune.contract.identity import validate_platform_key

SCHEMA_VERSION = 1
MODEL_BASE_URL = "https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans"

NVIDIA_H20_PLATFORM_KEY = "nvidia-h20"
NVIDIA_H20_MODEL_VERSION = "1.0.0"
NVIDIA_H20_REMOTE_FILENAME = "flagtune-xgb-nvidia-h20_v0.1.0.tar.gz"
NVIDIA_H20_PACKAGE_SHA256 = "b26b1057d3149df7de1e3bb91e6162bcb475709e41719bcf435f81ac3a2b8d4e"

# Add platforms and versions here. Each version accepts exactly one of
# ``filename`` or ``url``, together with the SHA-256 of the downloaded bytes.
PACKAGE_CATALOG: Mapping[str, Mapping[str, Mapping[str, Any]]] = {
    NVIDIA_H20_PLATFORM_KEY: {
        NVIDIA_H20_MODEL_VERSION: {
            "filename": NVIDIA_H20_REMOTE_FILENAME,
            "sha256": NVIDIA_H20_PACKAGE_SHA256,
        },
    },
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _package_url(metadata: Mapping[str, Any], base_url: str) -> str:
    keys = set(metadata)
    if keys not in ({"filename", "sha256"}, {"url", "sha256"}):
        raise ValueError("package metadata keys must be exactly filename+sha256 or url+sha256")
    if "url" in metadata:
        url = metadata["url"]
        if not isinstance(url, str):
            raise ValueError("package URL must be a string")
        url = url.strip()
    else:
        filename = metadata["filename"]
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("package filename must be a non-empty string")
        url = f"{base_url.rstrip('/')}/{filename.strip().lstrip('/')}"
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise ValueError(f"package URL must use HTTPS: {url!r}")
    return url


def build_manifest(
    catalog: Mapping[str, Mapping[str, Mapping[str, Any]]],
    base_url: str,
) -> dict[str, Any]:
    """Validate a variable-driven catalog and return Manifest schema 1."""
    if not isinstance(catalog, Mapping):
        raise ValueError("package catalog must be a mapping")
    packages = {}
    for platform_key, versions in catalog.items():
        normalized_platform = validate_platform_key(platform_key, "Manifest platform key")
        if normalized_platform != platform_key:
            raise ValueError(f"Manifest platform key must already be normalized: {platform_key!r}")
        if not isinstance(versions, Mapping):
            raise ValueError(f"versions for platform {platform_key!r} must be a mapping")
        generated_versions = {}
        for version, metadata in versions.items():
            validated_version = validate_model_version(version)
            if not isinstance(metadata, Mapping):
                raise ValueError(f"package metadata for {platform_key!r} version {version!r} must be a mapping")
            digest = metadata.get("sha256")
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"package SHA-256 for {platform_key!r} version {validated_version!r} "
                                 "must be 64 lowercase hexadecimal characters")
            generated_versions[validated_version] = {
                "url": _package_url(metadata, base_url),
                "sha256": digest,
            }
        packages[normalized_platform] = {"versions": generated_versions}
    return {"schema_version": SCHEMA_VERSION, "packages": packages}


def _configured_model_base_url() -> str:
    return os.environ.get("FLAGTUNE_MODEL_BASE_URL", "").strip() or MODEL_BASE_URL


def build_default_manifest(*, base_url: str | None = None) -> dict[str, Any]:
    """Build the bundled catalog using the configured or explicit base URL."""
    effective_base_url = _configured_model_base_url() if base_url is None else base_url
    return build_manifest(PACKAGE_CATALOG, effective_base_url)


def _default_manifest_path() -> Path:
    if "FLAGTUNE_LOCAL_MANIFEST" in os.environ:
        configured = os.environ["FLAGTUNE_LOCAL_MANIFEST"].strip()
        if not configured:
            raise ValueError("FLAGTUNE_LOCAL_MANIFEST is present but empty")
        return Path(configured).expanduser()
    cache = os.environ.get("FLAGTUNE_MODEL_CACHE")
    cache_root = Path(cache) if cache else Path.home() / ".flagtree" / "flagtune_models"
    return cache_root / "manifest.json"


def _temporary_manifest(path: Path, manifest: Mapping[str, Any]) -> Path:
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Atomically replace the generated Manifest with deterministic JSON."""
    temporary = _temporary_manifest(path, manifest)
    try:
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_manifest_if_missing(path: Path, manifest: Mapping[str, Any]) -> bool:
    """Atomically publish a Manifest without replacing an existing path."""
    temporary = _temporary_manifest(path, manifest)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def ensure_default_manifest(path: Path) -> bool:
    """Create the bundled default Manifest only when ``path`` is absent."""
    return write_manifest_if_missing(path, build_default_manifest())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Manifest output path; defaults to FLAGTUNE_LOCAL_MANIFEST or the model cache",
    )
    parser.add_argument(
        "--base-url",
        default=_configured_model_base_url(),
        help="HTTPS base URL joined with catalog filename entries",
    )
    args = parser.parse_args(argv)
    output = args.output.expanduser() if args.output is not None else _default_manifest_path()
    manifest = build_default_manifest(base_url=args.base_url)
    write_manifest(output, manifest)
    print(f"FlagTune local Manifest written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
