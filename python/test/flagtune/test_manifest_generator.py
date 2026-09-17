import json
import tarfile

import pytest

from triton.flagtune.training import manifest_generator


def test_default_catalog_uses_published_urls_and_package_versions():
    manifest = manifest_generator.build_manifest(
        manifest_generator.PACKAGE_CATALOG,
        manifest_generator.MODEL_BASE_URL,
    )

    assert manifest == {
        "schema_version": 1,
        "packages": {
            "hygon-bw": {
                "versions": {
                    "1.0.0": {
                        "url": ("https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/"
                                "flagtune-xgb-hygon-bw_v1.0.0.tar.gz"),
                        "sha256":
                        "5af5202f9354b9a09f34ff5c8e35ffce5868462def9a70729cb050a68bb0db33",
                    },
                },
            },
            "metax-c550": {
                "versions": {
                    "1.0.0": {
                        "url": ("https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/"
                                "flagtune-xgb-metax-c550_v1.0.0.tar.gz"),
                        "sha256":
                        "a1b770e1ed614606126f21b252b815270bda2f1796e688e285fe24a5642bc2b2",
                    },
                },
            },
            "mthreads-s5000": {
                "versions": {
                    "1.0.0": {
                        "url": ("https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/"
                                "flagtune-xgb-mthreads-s5000_v1.0.0.tar.gz"),
                        "sha256":
                        "7e8ab01abedded60c7d564b550e094fea2497e45b3cad92038b7dde64b8ad8d9",
                    },
                },
            },
            "nvidia-h20": {
                "versions": {
                    "1.0.0": {
                        "url": ("https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/"
                                "flagtune-xgb-nvidia-h20_v1.0.0.tar.gz"),
                        "sha256":
                        "1ffb2545402a8d0b92e95fcf747380aee2b52ed818cd00953a08e7dafc571759",
                    },
                },
            },
            "thead-zw810e": {
                "versions": {
                    "1.0.0": {
                        "url": ("https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/"
                                "flagtune-xgb-thead-zw810e_v1.0.0.tar.gz"),
                        "sha256":
                        "78858b99a2b2252385f2a8624aff4391d0235bfc270beeb07a8cb7e0c7174942",
                    },
                },
            },
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


def test_write_manifest_bundle_contains_only_manifest_json(tmp_path):
    manifest = {
        "schema_version": 1,
        "packages": {
            "nvidia-h20": {
                "versions": {
                    "1.0.0": {
                        "url": "https://example.invalid/model.tar.gz",
                        "sha256": "a" * 64,
                    },
                },
            },
        },
    }
    manifest_path, bundle_path = manifest_generator.write_manifest_bundle(
        manifest,
        tmp_path / "release" / "flagtune-manifest.tar.gz",
    )

    with tarfile.open(bundle_path, mode="r:gz") as archive:
        assert archive.getnames() == ["manifest.json"]
        extracted = archive.extractfile("manifest.json")
        assert extracted is not None
        assert json.loads(extracted.read()) == manifest
    assert json.loads(manifest_path.read_text()) == manifest


def test_build_default_manifest_uses_environment_base_url(monkeypatch):
    monkeypatch.setenv("FLAGTUNE_MODEL_BASE_URL", "https://mirror.example.com/flagtune/")

    manifest = manifest_generator.build_default_manifest()

    assert manifest["packages"]["nvidia-h20"]["versions"]["1.0.0"]["url"] == (
        "https://mirror.example.com/flagtune/flagtune-xgb-nvidia-h20_v1.0.0.tar.gz")


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
