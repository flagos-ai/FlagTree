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
"""Migrate legacy GPU-keyed XGBoost archives into platform packages.

Legacy ``gpu_key`` input is deliberately isolated here. Runtime and training
contracts remain strict consumers of ``platform_key`` and never fall back to
the old archive layout or configuration schema.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

from triton.flagtune._dependencies import require_xgboost
from triton.flagtune.contract.archive import (
    MODEL_ARCHIVE_NAME,
    ModelArchiveError,
    platform_package_name,
    read_model_archive_bytes,
    validate_model_version,
    write_model_archive,
    write_platform_package,
)
from triton.flagtune.contract.identity import ModelIdentity, make_platform_key, validate_identity_segment
from triton.flagtune.contract.operator_schema import (
    model_config_sha256,
    model_identity_from_config,
    parse_model_config,
)


def _load_legacy_yaml(payload: bytes) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("FlagTune model migration requires PyYAML") from exc
    try:
        value = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ModelArchiveError(f"invalid legacy FlagTune model YAML: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ModelArchiveError("legacy FlagTune model YAML must contain a mapping")
    return dict(value)


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ModelArchiveError(f"duplicate key in legacy training summary: {key!r}")
        result[key] = value
    return result


def _load_legacy_summary(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys)
    except ModelArchiveError:
        raise
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelArchiveError(f"invalid legacy training summary: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelArchiveError("legacy training summary must contain a JSON object")
    return value


def _migrate_summary_value(value: Any, platform_key: str) -> Any:
    if isinstance(value, dict):
        migrated = {}
        for key, item in value.items():
            target_key = "platform_key" if key == "gpu_key" else key
            if target_key in migrated or (key == "gpu_key" and "platform_key" in value):
                raise ModelArchiveError(f"legacy training summary has conflicting key {target_key!r}")
            migrated[target_key] = platform_key if key == "gpu_key" else _migrate_summary_value(item, platform_key)
        return migrated
    if isinstance(value, list):
        return [_migrate_summary_value(item, platform_key) for item in value]
    return value


def _migrate_config(config: Mapping[str, Any], platform_key: str, model_version: str) -> dict[str, Any]:
    migrated = dict(config)
    if migrated.get("format_version") != 5:
        raise ModelArchiveError("legacy model config.format_version must be 5")
    if "platform_key" in migrated or "gpu_key" not in migrated:
        raise ModelArchiveError("legacy model config must contain gpu_key and no platform_key")
    try:
        legacy_gpu_key = validate_identity_segment(migrated.pop("gpu_key"), "legacy model config.gpu_key")
    except (TypeError, ValueError) as exc:
        raise ModelArchiveError(f"invalid legacy gpu_key: {exc}") from exc

    gpu = migrated.get("gpu")
    if not isinstance(gpu, Mapping):
        raise ModelArchiveError("legacy model config.gpu must be a mapping")
    migrated_gpu = dict(gpu)
    if "platform_key" in migrated_gpu or "gpu_key" not in migrated_gpu:
        raise ModelArchiveError("legacy model config.gpu must contain gpu_key and no platform_key")
    try:
        nested_legacy_gpu_key = validate_identity_segment(migrated_gpu.pop("gpu_key"),
                                                          "legacy model config.gpu.gpu_key")
        derived_platform = make_platform_key(str(migrated_gpu["vendor"]), str(migrated_gpu["device_name"]))
        architecture = validate_identity_segment(migrated_gpu["architecture"], "legacy model config.gpu.architecture")
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelArchiveError(f"cannot derive platform from legacy GPU metadata: {exc}") from exc
    if architecture != "sm90":
        raise ModelArchiveError(f"legacy H20 migration requires architecture 'sm90', got {architecture!r}")
    expected_legacy_gpu_key = validate_identity_segment(f"{derived_platform}-{architecture}", "legacy gpu_key")
    if legacy_gpu_key != nested_legacy_gpu_key:
        raise ModelArchiveError("legacy gpu_key mismatch between model config root and GPU metadata")
    if legacy_gpu_key != expected_legacy_gpu_key:
        raise ModelArchiveError(f"legacy gpu_key {legacy_gpu_key!r} does not match GPU metadata "
                                f"{expected_legacy_gpu_key!r}")
    if derived_platform != platform_key:
        raise ModelArchiveError(f"legacy GPU metadata platform mismatch: {derived_platform!r} != {platform_key!r}")

    migrated["model_version"] = model_version
    migrated["platform_key"] = platform_key
    migrated_gpu["platform_key"] = platform_key
    migrated["gpu"] = migrated_gpu
    try:
        parse_model_config(migrated)
    except (TypeError, ValueError) as exc:
        raise ModelArchiveError(f"invalid migrated model config: {exc}") from exc
    return migrated


def _validate_legacy_summary(
    summary: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    feature_names: Sequence[str],
    legacy_digest: str,
) -> None:
    expected_features = list(feature_names)
    if summary.get("feature_cols") != expected_features:
        raise ModelArchiveError("legacy training summary feature_cols do not match model config")
    if "feature_count" in summary and summary["feature_count"] != len(expected_features):
        raise ModelArchiveError("legacy training summary feature_count does not match model config")
    for field in ("op_id", "variant", "model_version"):
        if field in summary and summary[field] != config.get(field):
            raise ModelArchiveError(f"legacy training summary {field} does not match model config")
    if summary.get("model_config_sha256") != legacy_digest:
        raise ModelArchiveError("legacy training summary config digest does not match model config")
    legacy_gpu_key = config.get("gpu_key")
    if "gpu_key" in summary and summary["gpu_key"] != legacy_gpu_key:
        raise ModelArchiveError("legacy training summary gpu_key does not match model config")
    summary_identity = summary.get("model_identity")
    if isinstance(summary_identity, Mapping):
        if summary_identity.get("gpu_key") != legacy_gpu_key:
            raise ModelArchiveError("legacy training summary model_identity.gpu_key does not match model config")
        if summary_identity.get("dtype_key") != config.get("dtype_key"):
            raise ModelArchiveError("legacy training summary model_identity.dtype_key does not match model config")


def migrate_model_archive(
    payload: bytes,
    *,
    platform_key: str,
    model_version: str,
) -> tuple[ModelIdentity, bytes]:
    """Rewrite one legacy child archive without fitting or changing its trees."""
    try:
        platform = validate_identity_segment(platform_key, "platform_key")
        version = validate_model_version(model_version)
    except (TypeError, ValueError) as exc:
        raise ModelArchiveError(f"invalid migration identity: {exc}") from exc
    members = read_model_archive_bytes(payload, source="legacy model archive")
    legacy_config = _load_legacy_yaml(members["flagtune_config.yaml"])
    config = _migrate_config(legacy_config, platform, version)
    variant = parse_model_config(config)
    legacy_digest = model_config_sha256(legacy_config)
    identity = model_identity_from_config(config)
    digest = model_config_sha256(config)
    legacy_summary = _load_legacy_summary(members["training_summary.json"])
    summary = _migrate_summary_value(legacy_summary, platform)
    summary["model_config_sha256"] = digest
    summary["model_version"] = version

    xgboost = require_xgboost("FlagTune model migration")
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("FlagTune model migration requires PyYAML") from exc

    with tempfile.TemporaryDirectory(prefix="flagtune-migrate-model-") as temporary_dir:
        temporary = Path(temporary_dir)
        legacy_model_path = temporary / "legacy_xgboost_ranker.json"
        migrated_model_path = temporary / "xgboost_ranker.json"
        legacy_model_path.write_bytes(members["xgboost_ranker.json"])
        model = xgboost.XGBRanker()
        try:
            model.load_model(str(legacy_model_path))
            booster = model.get_booster()
            stored_digest = booster.attr("flagtune_config_sha256")
            if stored_digest != legacy_digest:
                raise ModelArchiveError(f"legacy model config digest mismatch: {stored_digest!r} != {legacy_digest!r}")
            stored_features = list(booster.feature_names or [])
            expected_features = list(variant.feature_names)
            if stored_features and stored_features != expected_features:
                raise ModelArchiveError("legacy XGBoost feature order does not match model config")
            if int(booster.num_features()) != len(expected_features):
                raise ModelArchiveError("legacy XGBoost feature count does not match model config")
            _validate_legacy_summary(
                legacy_summary,
                config=legacy_config,
                feature_names=variant.feature_names,
                legacy_digest=legacy_digest,
            )
            booster.set_attr(flagtune_config_sha256=digest)
            model.save_model(str(migrated_model_path))
        except (OSError, TypeError, ValueError, xgboost.core.XGBoostError) as exc:
            raise ModelArchiveError(f"cannot migrate legacy XGBoost model: {exc}") from exc
        migrated_members = {
            "xgboost_ranker.json": migrated_model_path.read_bytes(),
            "flagtune_config.yaml": yaml.safe_dump(
                config,
                sort_keys=False,
                allow_unicode=True,
            ).encode("utf-8"),
            "training_summary.json": json.dumps(summary, indent=2, sort_keys=True).encode("utf-8"),
        }
        archive_path = write_model_archive(temporary / MODEL_ARCHIVE_NAME, migrated_members)
        migrated_payload = archive_path.read_bytes()
    return identity, migrated_payload


def migrate_platform_package(
    model_paths: Iterable[Path | str],
    *,
    platform_key: str,
    model_version: str,
    output_path: Path | str,
    required_identities: Sequence[ModelIdentity],
) -> Path:
    """Migrate legacy child archives and write their complete platform package."""
    try:
        platform = validate_identity_segment(platform_key, "platform_key")
        version = validate_model_version(model_version)
    except (TypeError, ValueError) as exc:
        raise ModelArchiveError(f"invalid migration identity: {exc}") from exc
    output = Path(output_path)
    expected_name = platform_package_name(platform, version)
    if output.name != expected_name:
        raise ModelArchiveError(f"platform package filename must be {expected_name!r}, got {output.name!r}")
    required = set(required_identities)
    with tempfile.TemporaryDirectory(prefix="flagtune-migrate-package-") as temporary_dir:
        temporary = Path(temporary_dir)
        archives: dict[ModelIdentity, Path] = {}
        for index, model_path in enumerate(model_paths):
            source = Path(model_path)
            identity, payload = migrate_model_archive(
                source.read_bytes(),
                platform_key=platform,
                model_version=version,
            )
            if identity in archives:
                raise ModelArchiveError(f"duplicate migrated model identity: {identity.artifact_key!r}")
            child_path = temporary / f"model-{index}.tar.gz"
            child_path.write_bytes(payload)
            archives[identity] = child_path

        actual = set(archives)
        missing = sorted(required - actual, key=lambda item: item.artifact_key)
        if missing:
            raise ModelArchiveError(
                f"platform package is missing required identities: {[item.artifact_key for item in missing]}")
        unexpected = sorted(actual - required, key=lambda item: item.artifact_key)
        if unexpected:
            raise ModelArchiveError(
                f"platform package has unexpected identities: {[item.artifact_key for item in unexpected]}")
        return write_platform_package(
            output,
            platform_key=platform,
            package_version=version,
            model_archives=archives,
            required_identities=required_identities,
        )


def _required_h20_identities(platform_key: str) -> tuple[ModelIdentity, ...]:
    h20_platform = make_platform_key("NVIDIA", "NVIDIA H20-3e")
    requested_platform = validate_identity_segment(platform_key, "platform_key")
    if requested_platform != h20_platform:
        raise ModelArchiveError(
            f"legacy H20 migration only supports platform {h20_platform!r}, got {requested_platform!r}")
    return tuple(
        ModelIdentity(h20_platform, "flaggems/mm", variant, "bf16-bf16-bf16")
        for variant in ("gemv", "general_tma", "splitk"))


def _validate_manifest_output(path: Path) -> tuple[bool, bytes]:
    if path.is_symlink():
        raise ModelArchiveError(f"Manifest output must not be a symlink: {path}")
    if not path.exists():
        return False, b""
    if not path.is_file():
        raise ModelArchiveError(f"Manifest output already exists and is not empty: {path}")
    payload = path.read_bytes()
    if not payload:
        return True, payload
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (ModelArchiveError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelArchiveError(f"Manifest output already exists and is not empty: {path}") from exc
    if (type(value) is not dict or set(value) != {"schema_version", "packages"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or type(value["packages"]) is not dict or value["packages"]):
        raise ModelArchiveError(f"Manifest output already exists and is not empty: {path}")
    return True, payload


@contextmanager
def _lock_manifest_output(path: Path):
    """Serialize cooperating publishers with a process-owned advisory lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        raise ModelArchiveError(f"cannot open Manifest publishing lock: {path}") from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ModelArchiveError(f"Manifest output is locked by another publisher: {path}") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
            yield
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _validate_package_url(url: str) -> str:
    package_url = url.strip()
    parsed = urlparse(package_url)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise ModelArchiveError(f"package URL must use HTTPS: {url!r}")
    return package_url


def _package_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_publishing_manifest(
    path: Path,
    *,
    platform_key: str,
    model_version: str,
    package_url: str,
    package_path: Path,
    expected_output_state: tuple[bool, bytes],
) -> None:
    manifest = {
        "schema_version": 1,
        "packages": {
            platform_key: {
                "versions": {
                    model_version: {
                        "url": package_url,
                        "sha256": _package_sha256(package_path),
                    },
                },
            },
        },
    }
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if expected_output_state[0]:
            try:
                current_output_state = _validate_manifest_output(path)
            except ModelArchiveError as exc:
                raise ModelArchiveError(f"Manifest output changed during publishing: {path}") from exc
            if current_output_state != expected_output_state:
                raise ModelArchiveError(f"Manifest output changed during publishing: {path}")
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise ModelArchiveError(f"Manifest output changed during publishing: {path}") from exc
            temporary.unlink()
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate legacy FlagTune XGBoost models")
    parser.add_argument("--platform-key", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True, action="append", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--package-url")
    args = parser.parse_args(argv)
    if (args.manifest_output is None) != (args.package_url is None):
        parser.error("--manifest-output and --package-url must be provided together")
    package_url = None
    if args.manifest_output is not None:
        package_url = _validate_package_url(args.package_url)
    publishing_lock = (_lock_manifest_output(args.manifest_output)
                       if args.manifest_output is not None else nullcontext())
    with publishing_lock:
        expected_output_state = (_validate_manifest_output(args.manifest_output)
                                 if args.manifest_output is not None else None)
        output = migrate_platform_package(
            args.model,
            platform_key=args.platform_key,
            model_version=args.model_version,
            output_path=args.output,
            required_identities=_required_h20_identities(args.platform_key),
        )
        if args.manifest_output is not None:
            _write_publishing_manifest(
                args.manifest_output,
                platform_key=args.platform_key,
                model_version=args.model_version,
                package_url=package_url,
                package_path=output,
                expected_output_state=expected_output_state,
            )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
