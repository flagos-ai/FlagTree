import json
import re
import tarfile
from io import BytesIO

import pytest

from triton.flagtune.contract.identity import ModelIdentityError
from triton.flagtune.runtime import model_sources

PLATFORM_KEY = "nvidia-h20"
ENTRY_1 = {
    "url": "https://example.invalid/nvidia-h20_v1.0.0.tar.gz",
    "sha256": "1" * 64,
}
ENTRY_2 = {
    "url": "https://example.invalid/nvidia-h20_v2.0.0.tar.gz",
    "sha256": "2" * 64,
}


@pytest.fixture(autouse=True)
def clean_manifest_environment(monkeypatch):
    for name in (
            "FLAGTUNE_LOCAL_MANIFEST",
            "FLAGTUNE_MODEL_CACHE",
            "FLAGTUNE_MODEL_BASE_URL",
            "FLAGTUNE_MANIFEST_URL",
            "FLAGTUNE_MANIFEST_TTL",
            "FLAGTUNE_MANIFEST_REFRESH",
            "FLAGTUNE_DISABLE_REMOTE",
    ):
        monkeypatch.delenv(name, raising=False)
    # Keep fixture-created cache manifests offline; the default URL is tested
    # explicitly below and should not make unrelated tests perform network I/O.
    monkeypatch.setenv("FLAGTUNE_MANIFEST_URL", "")


def test_default_manifest_url(monkeypatch):
    """Use the hosted FlagOS Manifest when no URL override is provided."""
    monkeypatch.delenv("FLAGTUNE_MANIFEST_URL", raising=False)

    assert model_sources._manifest_url() == ("https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/"
                                             "flagtune-xgb-manifest.tar.gz")


def manifest_with(entry, *, platform_key=PLATFORM_KEY):
    return {"schema_version": 1, "packages": {platform_key: entry}}


def write_manifest(path, entry, *, platform_key=PLATFORM_KEY):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest_with(entry, platform_key=platform_key)))
    return path


def install_cached_manifest(tmp_path, monkeypatch, entry, *, platform_key=PLATFORM_KEY):
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    return write_manifest(cache_root / "manifest.json", entry, platform_key=platform_key)


def install_local_manifest(tmp_path, monkeypatch, entry, *, platform_key=PLATFORM_KEY):
    path = write_manifest(tmp_path / "flagtune-local-manifest.json", entry, platform_key=platform_key)
    monkeypatch.setenv("FLAGTUNE_LOCAL_MANIFEST", str(path))
    return path


@pytest.mark.parametrize("source", ("cache", "explicit"))
def test_exact_pin_and_highest_semver_selection_ignore_latest(tmp_path, monkeypatch, source):
    entry = {
        "latest": "1.0.0",
        "versions": {"1.0.0": ENTRY_1, "2.0.0": ENTRY_2},
    }
    installer = install_cached_manifest if source == "cache" else install_local_manifest
    installer(tmp_path, monkeypatch, entry)

    highest = model_sources.resolve_package_info(PLATFORM_KEY)
    exact = model_sources.resolve_package_info(PLATFORM_KEY, version="1.0.0")

    assert highest == model_sources.RemotePackage("2.0.0", ENTRY_2["url"], "2" * 64)
    assert exact == model_sources.RemotePackage("1.0.0", ENTRY_1["url"], "1" * 64)
    assert model_sources.resolve_package_info(PLATFORM_KEY, version="3.0.0") is None


def test_explicit_manifest_precedes_cached_manifest(tmp_path, monkeypatch):
    install_cached_manifest(tmp_path, monkeypatch, {"versions": {"1.0.0": ENTRY_1}})
    install_local_manifest(tmp_path, monkeypatch, {"versions": {"2.0.0": ENTRY_2}})

    assert model_sources.resolve_package_info(PLATFORM_KEY) == model_sources.RemotePackage(
        "2.0.0",
        ENTRY_2["url"],
        "2" * 64,
    )


def test_local_manifest_accepts_relative_environment_path(tmp_path, monkeypatch):
    manifest = write_manifest(
        tmp_path / "flagtune-local-manifest.json",
        {"versions": {"1.0.0": ENTRY_1}},
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FLAGTUNE_LOCAL_MANIFEST", manifest.name)

    assert model_sources.resolve_package_info(PLATFORM_KEY) == model_sources.RemotePackage(
        "1.0.0",
        ENTRY_1["url"],
        "1" * 64,
    )


def test_local_manifest_rejects_empty_environment_value(monkeypatch):
    monkeypatch.setenv("FLAGTUNE_LOCAL_MANIFEST", "  ")

    with pytest.raises(model_sources.ManifestContractError, match="present but empty"):
        model_sources.resolve_package_info(PLATFORM_KEY)


def test_missing_cached_manifest_requires_remote_url(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    manifest_path = cache_root / "manifest.json"
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    monkeypatch.setenv("FLAGTUNE_MANIFEST_URL", "")

    with pytest.raises(model_sources.ManifestFetchError, match="FLAGTUNE_MANIFEST_URL is not configured"):
        model_sources.resolve_package_info(PLATFORM_KEY)

    assert not manifest_path.exists()


def test_remote_manifest_fetch_failure_is_reported(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    monkeypatch.setenv("FLAGTUNE_MANIFEST_URL", "https://example.invalid/flagtune-manifest.tar.gz")

    def fail_fetch():
        raise model_sources.ManifestFetchError("remote Manifest unavailable")

    monkeypatch.setattr(model_sources, "_fetch_remote_manifest", fail_fetch)

    with pytest.raises(model_sources.ManifestFetchError, match="remote Manifest unavailable"):
        model_sources.resolve_package_info(PLATFORM_KEY)

    assert not (cache_root / "manifest.json").exists()


def test_remote_manifest_bundle_contains_only_schema_validated_manifest():
    manifest = manifest_with({"versions": {"1.0.0": ENTRY_1}})
    manifest_bytes = json.dumps(manifest).encode("utf-8")
    payload = BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        info = tarfile.TarInfo("manifest.json")
        info.size = len(manifest_bytes)
        archive.addfile(info, BytesIO(manifest_bytes))

    extracted = model_sources._bundle_members(payload.getvalue(), "https://example.invalid/manifest.tar.gz")

    assert model_sources._read_manifest_bytes(extracted, "remote test") == manifest


@pytest.mark.parametrize(
    ("source", "case"),
    [
        ("explicit", "missing"),
        ("cache", "directory"),
        ("explicit", "directory"),
        ("cache", "invalid-json"),
        ("explicit", "invalid-json"),
    ],
)
def test_manifest_rejects_unreadable_input(tmp_path, monkeypatch, source, case):
    cache_root = tmp_path / "cache"
    path = cache_root / "manifest.json" if source == "cache" else tmp_path / "local-manifest.json"
    if case == "directory":
        path.mkdir(parents=True)
    elif case == "invalid-json":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not-json")

    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    if source == "explicit":
        monkeypatch.setenv("FLAGTUNE_LOCAL_MANIFEST", str(path))

    message = "regular file" if case != "invalid-json" else "valid JSON"
    with pytest.raises(model_sources.ManifestContractError, match=message):
        model_sources.resolve_package_info(PLATFORM_KEY)

    if source == "explicit" and case == "missing":
        assert not (cache_root / "manifest.json").exists()


@pytest.mark.parametrize(
    ("duplicate_key", "payload"),
    [
        (
            "packages",
            '{"schema_version":1,"packages":{},"packages":{}}',
        ),
        (
            "1.0.0",
            '{"schema_version":1,"packages":{"nvidia-h20":{"versions":{'
            '"1.0.0":{"url":"https://example.invalid/first.tar.gz","sha256":"'
            '1111111111111111111111111111111111111111111111111111111111111111"},'
            '"1.0.0":{"url":"https://example.invalid/second.tar.gz","sha256":"'
            '2222222222222222222222222222222222222222222222222222222222222222"}'
            '}}}}',
        ),
        (
            "url",
            '{"schema_version":1,"packages":{"nvidia-h20":{"versions":{'
            '"1.0.0":{"url":"https://example.invalid/first.tar.gz",'
            '"url":"https://example.invalid/second.tar.gz",'
            '"sha256":"1111111111111111111111111111111111111111111111111111111111111111"}'
            '}}}}',
        ),
    ],
)
def test_manifest_rejects_duplicate_json_keys_at_any_object_level(
    tmp_path,
    monkeypatch,
    duplicate_key,
    payload,
):
    cache_root = tmp_path / "cache"
    path = cache_root / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(payload)
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))

    with pytest.raises(
            model_sources.ManifestContractError,
            match=rf"duplicate JSON key.*{re.escape(duplicate_key)}",
    ):
        model_sources.resolve_package_info(PLATFORM_KEY)


@pytest.mark.parametrize("source", ("cache", "explicit"))
def test_manifest_rejects_symlink(tmp_path, monkeypatch, source):
    target = write_manifest(
        tmp_path / "target.json",
        {"versions": {"1.0.0": ENTRY_1}},
    )
    cache_root = tmp_path / "cache"
    link = cache_root / "manifest.json" if source == "cache" else tmp_path / "manifest-link.json"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    if source == "explicit":
        monkeypatch.setenv("FLAGTUNE_LOCAL_MANIFEST", str(link))

    with pytest.raises(model_sources.ManifestContractError, match="symlink"):
        model_sources.resolve_package_info(PLATFORM_KEY)


@pytest.mark.parametrize("source", ("cache", "explicit"))
@pytest.mark.parametrize("location", ("root", "package", "metadata"))
def test_manifest_rejects_unknown_fields(tmp_path, monkeypatch, source, location):
    manifest = manifest_with({"versions": {"1.0.0": dict(ENTRY_1)}})
    if location == "root":
        manifest["extra"] = True
    elif location == "package":
        manifest["packages"][PLATFORM_KEY]["extra"] = True
    else:
        manifest["packages"][PLATFORM_KEY]["versions"]["1.0.0"]["extra"] = True

    cache_root = tmp_path / "cache"
    path = cache_root / "manifest.json" if source == "cache" else tmp_path / "local-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest))
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    if source == "explicit":
        monkeypatch.setenv("FLAGTUNE_LOCAL_MANIFEST", str(path))

    with pytest.raises(model_sources.ManifestContractError, match="schema 1"):
        model_sources.resolve_package_info(PLATFORM_KEY)


@pytest.mark.parametrize("source", ("cache", "explicit"))
def test_manifest_missing_platform_returns_none(tmp_path, monkeypatch, source):
    installer = install_cached_manifest if source == "cache" else install_local_manifest
    installer(
        tmp_path,
        monkeypatch,
        {
            "versions": {
                "1.0.0": {
                    "url": "https://example.invalid/nvidia-a100_v1.0.0.tar.gz",
                    "sha256": "1" * 64,
                },
            },
        },
        platform_key="nvidia-a100",
    )

    assert model_sources.resolve_package_info(PLATFORM_KEY) is None


@pytest.mark.parametrize(
    "metadata",
    [
        {"file": "../nvidia-h20_v1.0.0.tar.gz", "sha256": "a" * 64},
        {"file": "/tmp/nvidia-h20_v1.0.0.tar.gz", "sha256": "a" * 64},
        {"file": "models/nvidia-h20_v1.0.0.tar.gz", "sha256": "a" * 64},
        {"file": r"models\nvidia-h20_v1.0.0.tar.gz", "sha256": "a" * 64},
        {"file": "nvidia-h20_v2.0.0.tar.gz", "sha256": "a" * 64},
        {"file": "nvidia-h20_v1.0.0.tar.gz", "sha256": "A" * 64},
    ],
)
def test_manifest_rejects_file_metadata(tmp_path, monkeypatch, metadata):
    install_cached_manifest(tmp_path, monkeypatch, {"versions": {"1.0.0": metadata}})

    with pytest.raises(model_sources.ManifestContractError, match="schema 1"):
        model_sources.resolve_package_info(PLATFORM_KEY)


def test_platform_key_is_normalized_and_validated(tmp_path, monkeypatch):
    install_cached_manifest(tmp_path, monkeypatch, {"versions": {"1.0.0": ENTRY_1}})

    assert model_sources.resolve_package_info(" NVIDIA-H20 ") == model_sources.RemotePackage(
        "1.0.0",
        ENTRY_1["url"],
        "1" * 64,
    )
    with pytest.raises(ModelIdentityError):
        model_sources.resolve_package_info("../nvidia-h20")


@pytest.mark.parametrize(
    ("metadata", "accepted"),
    [
        ({"url": "https://example.invalid/nvidia-h20_v1.0.0.tar.gz", "sha256": "a" * 64}, True),
        ({"url": "https://example.invalid/flagtune-xgb-nvidia-h20_v0.1.0.tar.gz", "sha256": "a" * 64}, True),
        ({"url": "https://example.invalid/nvidia-h20_v2.0.0.tar.gz", "sha256": "a" * 64}, True),
        ({"url": "https://example.invalid/nvidia-h20_v1.0.0.tgz", "sha256": "a" * 64}, True),
        ({"url": "https://example.invalid/other_v1.0.0.tar.gz", "sha256": "a" * 64}, True),
        ({"url": "http://example.invalid/nvidia-h20_v1.0.0.tar.gz", "sha256": "a" * 64}, False),
        ({"url": "https:///nvidia-h20_v1.0.0.tar.gz", "sha256": "a" * 64}, False),
        ({"url": "https://example.invalid/nvidia-h20_v1.0.0.tar.gz"}, False),
        ({"url": "https://example.invalid/nvidia-h20_v1.0.0.tar.gz", "sha256": "A" * 64}, False),
        ({"url": "https://example.invalid/nvidia-h20_v1.0.0.tar.gz", "sha256": "a" * 63}, False),
    ],
)
def test_package_metadata_requires_https_and_lowercase_sha(tmp_path, monkeypatch, metadata, accepted):
    install_cached_manifest(tmp_path, monkeypatch, {"versions": {"1.0.0": metadata}})

    if accepted:
        assert model_sources.resolve_package_info(PLATFORM_KEY) is not None
    else:
        with pytest.raises(model_sources.ManifestContractError, match="schema 1"):
            model_sources.resolve_package_info(PLATFORM_KEY)


def test_manifest_accepts_transport_filenames_independent_of_package_identity(tmp_path, monkeypatch):
    entry = {
        "versions": {
            "1.0.0": {**ENTRY_1, "url": "https://example.invalid/download"},
            "2.0.0": {
                **ENTRY_2,
                "url": "https://example.invalid/flagtune-xgb-nvidia-h20_v2.0.0.tar.gz",
            },
        },
    }
    install_cached_manifest(tmp_path, monkeypatch, entry)

    assert model_sources.resolve_package_info(PLATFORM_KEY) == model_sources.RemotePackage(
        "2.0.0",
        entry["versions"]["2.0.0"]["url"],
        ENTRY_2["sha256"],
    )
