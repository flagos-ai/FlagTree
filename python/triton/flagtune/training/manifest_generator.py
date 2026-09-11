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
"""Generate a local or release FlagTune download Manifest.

Edit the variables in the catalog section to publish more platforms or
versions. A version may use ``filename`` to join against ``MODEL_BASE_URL`` or
an explicit ``url`` to select another HTTPS endpoint.

Run from a source checkout with::

    PYTHONPATH=python python -m triton.flagtune.training.manifest_generator

For a release bundle, provide a catalog whose entries contain a local package
``path`` and run::

    PYTHONPATH=python python -m triton.flagtune.training.manifest_generator \
        --catalog release/catalog.json \
        --base-url https://models.example.com/flagtune \
        --bundle-output release/flagtune-manifest.tar.gz

The output path is ``FLAGTUNE_LOCAL_MANIFEST`` when configured, otherwise
``$FLAGTUNE_MODEL_CACHE/manifest.json``. Neither timestamps nor a ``latest``
field are emitted; runtime selection computes the highest strict SemVer key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from triton.flagtune.contract.archive import validate_model_version
from triton.flagtune.contract.identity import validate_platform_key

SCHEMA_VERSION = 1
MODEL_BASE_URL = "https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans"

MODEL_VERSION = "1.0.0"

# Add platforms and versions here for the default publishing catalog. Each version accepts exactly one of
# ``filename`` or ``url``, together with the SHA-256 of the downloaded bytes.
PACKAGE_CATALOG: Mapping[str, Mapping[str, Mapping[str, Any]]] = {
    "hygon-bw": {
        MODEL_VERSION: {
            "filename": "flagtune-xgb-hygon-bw_v1.0.0.tar.gz",
            "sha256": "5af5202f9354b9a09f34ff5c8e35ffce5868462def9a70729cb050a68bb0db33",
        },
    },
    "metax-c550": {
        MODEL_VERSION: {
            "filename": "flagtune-xgb-metax-c550_v1.0.0.tar.gz",
            "sha256": "a1b770e1ed614606126f21b252b815270bda2f1796e688e285fe24a5642bc2b2",
        },
    },
    "mthreads-s5000": {
        MODEL_VERSION: {
            "filename": "flagtune-xgb-mthreads-s5000_v1.0.0.tar.gz",
            "sha256": "7e8ab01abedded60c7d564b550e094fea2497e45b3cad92038b7dde64b8ad8d9",
        },
    },
    "nvidia-h20": {
        MODEL_VERSION: {
            "filename": "flagtune-xgb-nvidia-h20_v1.0.0.tar.gz",
            "sha256": "1ffb2545402a8d0b92e95fcf747380aee2b52ed818cd00953a08e7dafc571759",
        },
    },
    "thead-zw810e": {
        MODEL_VERSION: {
            "filename": "flagtune-xgb-thead-zw810e_v1.0.0.tar.gz",
            "sha256": "78858b99a2b2252385f2a8624aff4391d0235bfc270beeb07a8cb7e0c7174942",
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
    """Build the default publishing catalog using the configured or explicit base URL."""
    effective_base_url = _configured_model_base_url() if base_url is None else base_url
    return build_manifest(PACKAGE_CATALOG, effective_base_url)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read catalog JSON {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot read model package {path}: {exc}") from exc
    return digest.hexdigest()


def build_release_manifest(
    catalog: Mapping[str, Mapping[str, Mapping[str, Any]]],
    base_url: str,
    *,
    catalog_root: Path | None = None,
) -> dict[str, Any]:
    """Build a Manifest while calculating digests from local model archives.

    Release catalogs use the normal ``filename``/``url`` fields plus a local
    ``path`` field.  The path is consumed by the publisher and never appears
    in the runtime Manifest.
    """
    if not isinstance(catalog, Mapping):
        raise ValueError("release catalog must be a mapping")
    root = catalog_root or Path.cwd()
    digest_catalog = {}
    for platform_key, versions in catalog.items():
        if not isinstance(versions, Mapping):
            raise ValueError(f"versions for platform {platform_key!r} must be a mapping")
        digest_versions = {}
        for version, metadata in versions.items():
            if not isinstance(metadata, Mapping):
                raise ValueError(f"package metadata for {platform_key!r} version {version!r} must be a mapping")
            package_path = metadata.get("path")
            if not isinstance(package_path, str) or not package_path.strip():
                raise ValueError(f"release metadata for {platform_key!r} version {version!r} requires a non-empty path")
            path = Path(package_path).expanduser()
            if not path.is_absolute():
                path = root / path
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"release model package must be a regular file: {path}")
            digest_metadata = {key: metadata[key] for key in ("filename", "url") if key in metadata}
            if not digest_metadata:
                digest_metadata["filename"] = path.name
            digest_metadata["sha256"] = _sha256_file(path)
            digest_versions[version] = digest_metadata
        digest_catalog[platform_key] = digest_versions
    return build_manifest(digest_catalog, base_url)


def _manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_manifest_bundle(
    manifest: Mapping[str, Any],
    bundle_path: Path,
) -> tuple[Path, Path]:
    """Write Manifest JSON and a deterministic tar.gz bundle."""
    manifest_bytes = _manifest_bytes(manifest)
    bundle_path = bundle_path.expanduser()
    manifest_path = bundle_path.parent / "manifest.json"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest_bytes)
    with bundle_path.open("wb") as output:
        import gzip

        with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                info = tarfile.TarInfo("manifest.json")
                info.size = len(manifest_bytes)
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, BytesIO(manifest_bytes))
    return manifest_path, bundle_path


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
    payload = _manifest_bytes(manifest)
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
    parser.add_argument(
        "--catalog",
        type=Path,
        help="Release catalog JSON; entries must include a local package path",
    )
    parser.add_argument(
        "--bundle-output",
        type=Path,
        help="Write manifest.json and this manifest tar.gz bundle",
    )
    args = parser.parse_args(argv)
    if args.catalog is None:
        manifest = build_default_manifest(base_url=args.base_url)
    else:
        catalog_path = args.catalog.expanduser()
        catalog = _load_json(catalog_path)
        manifest = build_release_manifest(
            catalog,
            args.base_url,
            catalog_root=catalog_path.parent,
        )
    if args.bundle_output is not None:
        manifest_path, bundle_path = write_manifest_bundle(
            manifest,
            args.bundle_output,
        )
        print(f"FlagTune Manifest written to {manifest_path}")
        print(f"FlagTune Manifest bundle written to {bundle_path}")
    else:
        output = args.output.expanduser() if args.output is not None else _default_manifest_path()
        write_manifest(output, manifest)
        print(f"FlagTune local Manifest written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
