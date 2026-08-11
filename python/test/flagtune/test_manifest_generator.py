import json

import pytest

from triton.flagtune.training import manifest_generator


def test_default_h20_catalog_uses_published_url_and_package_version():
    manifest = manifest_generator.build_manifest(
        manifest_generator.PACKAGE_CATALOG,
        manifest_generator.MODEL_BASE_URL,
    )

    assert manifest["packages"]["nvidia-h20"]["versions"] == {
        "1.0.0": {
            "url": ("https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/"
                    "flagtune-xgb-nvidia-h20_v0.1.0.tar.gz"),
            "sha256": "b26b1057d3149df7de1e3bb91e6162bcb475709e41719bcf435f81ac3a2b8d4e",
        },
    }


def test_build_manifest_supports_multiple_platforms_versions_and_url_override():
    catalog = {
        "nvidia-h20": {
            "1.0.0": {
                "filename": "flagtune-xgb-nvidia-h20_v0.1.0.tar.gz",
                "sha256": "1" * 64,
            },
            "2.0.0": {
                "url": "https://mirror.example.com/h20-v2.tar.gz",
                "sha256": "2" * 64,
            },
        },
        "amd-mi300x": {
            "1.5.0": {
                "filename": "mi300x-v1.5.0.tar.gz",
                "sha256": "3" * 64,
            },
        },
    }

    manifest = manifest_generator.build_manifest(catalog, "https://models.example.com/flagtune/")

    assert manifest == {
        "schema_version": 1,
        "packages": {
            "nvidia-h20": {
                "versions": {
                    "1.0.0": {
                        "url": "https://models.example.com/flagtune/flagtune-xgb-nvidia-h20_v0.1.0.tar.gz",
                        "sha256": "1" * 64,
                    },
                    "2.0.0": {
                        "url": "https://mirror.example.com/h20-v2.tar.gz",
                        "sha256": "2" * 64,
                    },
                },
            },
            "amd-mi300x": {
                "versions": {
                    "1.5.0": {
                        "url": "https://models.example.com/flagtune/mi300x-v1.5.0.tar.gz",
                        "sha256": "3" * 64,
                    },
                },
            },
        },
    }


@pytest.mark.parametrize(
    ("catalog", "message"),
    [
        ({"NVIDIA-H20": {"1.0.0": {"filename": "model.tar.gz", "sha256": "a" * 64}}}, "platform"),
        ({"nvidia-h20": {"latest": {"filename": "model.tar.gz", "sha256": "a" * 64}}}, "version"),
        ({"nvidia-h20": {"1.0.0": {"url": "http://example.com/model.tar.gz", "sha256": "a" * 64}}}, "HTTPS"),
        ({"nvidia-h20": {"1.0.0": {"filename": "model.tar.gz", "sha256": "A" * 64}}}, "SHA-256"),
        ({"nvidia-h20": {"1.0.0": {"filename": "model.tar.gz", "sha256": "a" * 64, "extra": 1}}}, "keys"),
    ],
)
def test_build_manifest_rejects_invalid_catalog(catalog, message):
    with pytest.raises(ValueError, match=message):
        manifest_generator.build_manifest(catalog, "https://models.example.com/flagtune")


def test_main_writes_deterministic_manifest_to_model_cache(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("FLAGTUNE_LOCAL_MANIFEST", raising=False)
    monkeypatch.setattr(
        manifest_generator,
        "PACKAGE_CATALOG",
        {
            "nvidia-h20": {
                "1.0.0": {
                    "filename": "flagtune-xgb-nvidia-h20_v0.1.0.tar.gz",
                    "sha256": "a" * 64,
                },
            },
        },
    )

    assert manifest_generator.main(["--base-url", "https://download.example.com/models"]) == 0

    output = tmp_path / "cache" / "manifest.json"
    expected = {
        "schema_version": 1,
        "packages": {
            "nvidia-h20": {
                "versions": {
                    "1.0.0": {
                        "url": "https://download.example.com/models/flagtune-xgb-nvidia-h20_v0.1.0.tar.gz",
                        "sha256": "a" * 64,
                    },
                },
            },
        },
    }
    assert json.loads(output.read_text()) == expected
    assert output.read_text() == json.dumps(expected, indent=2, sort_keys=True) + "\n"
    assert str(output) in capsys.readouterr().out


def test_write_manifest_if_missing_never_replaces_existing_content(tmp_path):
    output = tmp_path / "cache" / "manifest.json"
    first = {"schema_version": 1, "packages": {"first": {}}}
    second = {"schema_version": 1, "packages": {"second": {}}}

    assert manifest_generator.write_manifest_if_missing(output, first) is True
    first_bytes = output.read_bytes()
    assert manifest_generator.write_manifest_if_missing(output, second) is False

    assert output.read_bytes() == first_bytes
    assert json.loads(first_bytes) == first
    assert not list(output.parent.glob(".manifest.json.*.tmp"))


def test_build_default_manifest_uses_environment_base_url(monkeypatch):
    monkeypatch.setenv("FLAGTUNE_MODEL_BASE_URL", "https://mirror.example.com/flagtune/")

    manifest = manifest_generator.build_default_manifest()

    assert manifest["packages"]["nvidia-h20"]["versions"]["1.0.0"]["url"] == (
        "https://mirror.example.com/flagtune/flagtune-xgb-nvidia-h20_v0.1.0.tar.gz")


def test_local_manifest_environment_controls_generator_output(tmp_path, monkeypatch):
    output = tmp_path / "deployment" / "manifest.json"
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path / "ignored-cache"))
    monkeypatch.setenv("FLAGTUNE_LOCAL_MANIFEST", str(output))
    monkeypatch.setattr(
        manifest_generator,
        "PACKAGE_CATALOG",
        {
            "nvidia-h20": {
                "1.0.0": {
                    "url": "https://download.example.com/model.tar.gz",
                    "sha256": "a" * 64,
                },
            },
        },
    )

    assert manifest_generator.main([]) == 0

    assert output.is_file()
    assert not (tmp_path / "ignored-cache" / "manifest.json").exists()


def test_explicit_output_precedes_local_manifest_and_model_cache(tmp_path, monkeypatch):
    output = tmp_path / "explicit" / "manifest.json"
    local_output = tmp_path / "local" / "manifest.json"
    cache_output = tmp_path / "cache" / "manifest.json"
    monkeypatch.setenv("FLAGTUNE_LOCAL_MANIFEST", str(local_output))
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_output.parent))
    monkeypatch.setattr(
        manifest_generator,
        "PACKAGE_CATALOG",
        {
            "nvidia-h20": {
                "1.0.0": {
                    "url": "https://download.example.com/model.tar.gz",
                    "sha256": "a" * 64,
                },
            },
        },
    )

    assert manifest_generator.main(["--output", str(output)]) == 0

    assert output.is_file()
    assert not local_output.exists()
    assert not cache_output.exists()
