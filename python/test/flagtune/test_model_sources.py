import json
import re

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
    for name in ("FLAGTUNE_LOCAL_MANIFEST", "FLAGTUNE_MODEL_CACHE", "FLAGTUNE_MODEL_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


def manifest_with(entry, *, platform_key=PLATFORM_KEY):
    return {"schema_version": 1, "packages": {platform_key: entry}}


def write_manifest(path, entry, *, platform_key=PLATFORM_KEY):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest_with(entry, platform_key=platform_key)))
    return path


def install_default_manifest(tmp_path, monkeypatch, entry, *, platform_key=PLATFORM_KEY):
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    return write_manifest(cache_root / "manifest.json", entry, platform_key=platform_key)


def install_local_manifest(tmp_path, monkeypatch, entry, *, platform_key=PLATFORM_KEY):
    path = write_manifest(tmp_path / "flagtune-local-manifest.json", entry, platform_key=platform_key)
    monkeypatch.setenv("FLAGTUNE_LOCAL_MANIFEST", str(path))
    return path


@pytest.mark.parametrize("source", ("default", "explicit"))
def test_exact_pin_and_highest_semver_selection_ignore_latest(tmp_path, monkeypatch, source):
    entry = {
        "latest": "1.0.0",
        "versions": {"1.0.0": ENTRY_1, "2.0.0": ENTRY_2},
    }
    installer = install_default_manifest if source == "default" else install_local_manifest
    installer(tmp_path, monkeypatch, entry)

    highest = model_sources.resolve_package_info(PLATFORM_KEY)
    exact = model_sources.resolve_package_info(PLATFORM_KEY, version="1.0.0")

    assert highest == model_sources.RemotePackage("2.0.0", ENTRY_2["url"], "2" * 64)
    assert exact == model_sources.RemotePackage("1.0.0", ENTRY_1["url"], "1" * 64)
    assert model_sources.resolve_package_info(PLATFORM_KEY, version="3.0.0") is None


def test_explicit_manifest_precedes_default_cache_manifest(tmp_path, monkeypatch):
    install_default_manifest(tmp_path, monkeypatch, {"versions": {"1.0.0": ENTRY_1}})
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


def test_missing_default_manifest_is_generated_from_bundled_catalog(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    manifest_path = cache_root / "manifest.json"
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))

    package = model_sources.resolve_package_info(PLATFORM_KEY)

    assert package == model_sources.RemotePackage(
        "1.0.0",
        "https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtune-xgb-nvidia-h20_v0.1.0.tar.gz",
        "b26b1057d3149df7de1e3bb91e6162bcb475709e41719bcf435f81ac3a2b8d4e",
    )
    assert json.loads(manifest_path.read_text())["packages"][PLATFORM_KEY]["versions"]["1.0.0"] == {
        "url": package.url,
        "sha256": package.sha256,
    }
    assert not list(cache_root.glob(".manifest.json.*.tmp"))


def test_default_manifest_generation_can_be_disabled(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    manifest_path = cache_root / "manifest.json"
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))

    with pytest.raises(model_sources.ManifestContractError, match="regular file"):
        model_sources.resolve_package_info(PLATFORM_KEY, generate_default=False)

    assert not manifest_path.exists()


@pytest.mark.parametrize(
    ("source", "case"),
    [
        ("explicit", "missing"),
        ("default", "directory"),
        ("explicit", "directory"),
        ("default", "invalid-json"),
        ("explicit", "invalid-json"),
    ],
)
def test_manifest_rejects_unreadable_input(tmp_path, monkeypatch, source, case):
    cache_root = tmp_path / "cache"
    path = cache_root / "manifest.json" if source == "default" else tmp_path / "local-manifest.json"
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


@pytest.mark.parametrize("source", ("default", "explicit"))
def test_manifest_rejects_symlink(tmp_path, monkeypatch, source):
    target = write_manifest(
        tmp_path / "target.json",
        {"versions": {"1.0.0": ENTRY_1}},
    )
    cache_root = tmp_path / "cache"
    link = cache_root / "manifest.json" if source == "default" else tmp_path / "manifest-link.json"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    if source == "explicit":
        monkeypatch.setenv("FLAGTUNE_LOCAL_MANIFEST", str(link))

    with pytest.raises(model_sources.ManifestContractError, match="symlink"):
        model_sources.resolve_package_info(PLATFORM_KEY)


@pytest.mark.parametrize("source", ("default", "explicit"))
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
    path = cache_root / "manifest.json" if source == "default" else tmp_path / "local-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest))
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    if source == "explicit":
        monkeypatch.setenv("FLAGTUNE_LOCAL_MANIFEST", str(path))

    with pytest.raises(model_sources.ManifestContractError, match="schema 1"):
        model_sources.resolve_package_info(PLATFORM_KEY)


@pytest.mark.parametrize("source", ("default", "explicit"))
def test_manifest_missing_platform_returns_none(tmp_path, monkeypatch, source):
    installer = install_default_manifest if source == "default" else install_local_manifest
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
    install_default_manifest(tmp_path, monkeypatch, {"versions": {"1.0.0": metadata}})

    with pytest.raises(model_sources.ManifestContractError, match="schema 1"):
        model_sources.resolve_package_info(PLATFORM_KEY)


def test_platform_key_is_normalized_and_validated(tmp_path, monkeypatch):
    install_default_manifest(tmp_path, monkeypatch, {"versions": {"1.0.0": ENTRY_1}})

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
    install_default_manifest(tmp_path, monkeypatch, {"versions": {"1.0.0": metadata}})

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
    install_default_manifest(tmp_path, monkeypatch, entry)

    assert model_sources.resolve_package_info(PLATFORM_KEY) == model_sources.RemotePackage(
        "2.0.0",
        entry["versions"]["2.0.0"]["url"],
        ENTRY_2["sha256"],
    )
