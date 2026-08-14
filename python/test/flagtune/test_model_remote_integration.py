"""Exercise the published H20 model package over its public HTTPS endpoint."""

from __future__ import annotations

import hashlib
import json
import os

import pytest

from triton.flagtune.contract.archive import read_platform_package
from triton.flagtune.runtime.model_loader import FlagTuneModelManager

RUN_REMOTE_ENV = "FLAGTUNE_RUN_REMOTE_INTEGRATION"
PLATFORM_KEY = "nvidia-h20"
MODEL_VERSION = "1.0.0"
REMOTE_URL = ("https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/"
              "flagtune-xgb-nvidia-h20_v0.1.0.tar.gz")
PACKAGE_SHA256 = "b26b1057d3149df7de1e3bb91e6162bcb475709e41719bcf435f81ac3a2b8d4e"
EXPECTED_MODELS = {
    "nvidia-h20/flaggems/mm/gemv/bf16-bf16-bf16": {
        "path": "flaggems/mm/gemv/bf16-bf16-bf16/model.tar.gz",
    },
    "nvidia-h20/flaggems/mm/general_tma/bf16-bf16-bf16": {
        "path": "flaggems/mm/general_tma/bf16-bf16-bf16/model.tar.gz",
    },
    "nvidia-h20/flaggems/mm/splitk/bf16-bf16-bf16": {
        "path": "flaggems/mm/splitk/bf16-bf16-bf16/model.tar.gz",
    },
}

pytestmark = [
    pytest.mark.remote_integration,
    pytest.mark.skipif(
        os.environ.get(RUN_REMOTE_ENV) != "1",
        reason=f"set {RUN_REMOTE_ENV}=1 to exercise the published model endpoint",
    ),
]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_ksyun_h20_package_download_validation_and_cache_reuse(tmp_path, monkeypatch):
    cache_root = tmp_path / "model-cache"
    for name in (
            "FLAGTUNE_DISABLE_REMOTE",
            "FLAGTUNE_LOCAL_MANIFEST",
            "FLAGTUNE_MODEL_BASE_URL",
            "FLAGTUNE_MODEL_DIR",
            "FLAGTUNE_MODEL_DOWNLOAD_LATEST",
            "FLAGTUNE_MODEL_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))

    # Archive parsing below validates every child config without requiring the
    # optional XGBoost runtime to be installed in the integration-test host.
    monkeypatch.setattr(FlagTuneModelManager, "_validate_bundle_members", lambda *_args, **_kwargs: None)
    downloaded = FlagTuneModelManager().resolve(
        "flaggems/mm",
        "gemv",
        platform_key=PLATFORM_KEY,
        dtype_key="bf16-bf16-bf16",
    )
    expected = (cache_root / "packages" / PLATFORM_KEY / MODEL_VERSION / f"{PLATFORM_KEY}_v{MODEL_VERSION}.tar.gz")

    assert downloaded == expected
    assert downloaded.is_file()
    assert _sha256(downloaded) == PACKAGE_SHA256

    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["packages"][PLATFORM_KEY]["versions"][MODEL_VERSION] == {
        "url": REMOTE_URL,
        "sha256": PACKAGE_SHA256,
    }

    parsed = read_platform_package(
        downloaded,
        expected_platform_key=PLATFORM_KEY,
        expected_version=MODEL_VERSION,
    )
    assert parsed.platform_key == PLATFORM_KEY
    assert parsed.package_version == MODEL_VERSION
    assert parsed.models == EXPECTED_MODELS

    monkeypatch.setenv("FLAGTUNE_DISABLE_REMOTE", "1")
    reused = FlagTuneModelManager().resolve(
        "flaggems/mm",
        "gemv",
        platform_key=PLATFORM_KEY,
        dtype_key="bf16-bf16-bf16",
    )
    assert reused == downloaded
    assert not list(cache_root.rglob("*.tmp"))
