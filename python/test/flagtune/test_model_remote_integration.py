"""Exercise the published model packages over their public HTTPS endpoints."""

from __future__ import annotations

import hashlib
import json
import os
from itertools import islice
from pathlib import Path

import pytest

from triton.flagtune.contract.archive import read_platform_package
from triton.flagtune.contract.operator_schema import _MISSING
from triton.flagtune.runtime import model_sources
from triton.flagtune.runtime.model_loader import FlagTuneModelManager
from triton.flagtune.runtime.proposer import make_config_proposer

RUN_REMOTE_ENV = "FLAGTUNE_RUN_REMOTE_INTEGRATION"
MODEL_VERSION = "1.0.0"
THEAD_MM_VERSION = "1.1.0"
REMOTE_BASE_URL = "https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans"
REMOTE_MANIFEST_URL = os.environ.get("FLAGTUNE_TEST_MANIFEST_URL", model_sources._DEFAULT_MANIFEST_URL)
PUBLISHED_PACKAGES = (
    (
        "hygon-bw",
        "flagtune-xgb-hygon-bw_v1.0.0.tar.gz",
        "5af5202f9354b9a09f34ff5c8e35ffce5868462def9a70729cb050a68bb0db33",
    ),
    (
        "metax-c550",
        "flagtune-xgb-metax-c550_v1.0.0.tar.gz",
        "a1b770e1ed614606126f21b252b815270bda2f1796e688e285fe24a5642bc2b2",
    ),
    (
        "mthreads-s5000",
        "flagtune-xgb-mthreads-s5000_v1.0.0.tar.gz",
        "7e8ab01abedded60c7d564b550e094fea2497e45b3cad92038b7dde64b8ad8d9",
    ),
    (
        "nvidia-h20",
        "flagtune-xgb-nvidia-h20_v1.0.0.tar.gz",
        "1ffb2545402a8d0b92e95fcf747380aee2b52ed818cd00953a08e7dafc571759",
    ),
    (
        "thead-zw810e",
        "flagtune-xgb-thead-zw810e_v1.0.0.tar.gz",
        "78858b99a2b2252385f2a8624aff4391d0235bfc270beeb07a8cb7e0c7174942",
    ),
    (
        "thead-zw810e",
        "flagtune-xgb-thead-zw810e_v1.1.0.tar.gz",
        "0a089ab5cba39867a0ce57992a65e7991e33ca26549dcf893e736bcff3feb0cc",
    ),
)

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


def _prediction_inputs(variant):
    """Cover dense, GEMV and split-K predicates without requiring a GPU."""
    required = {field.name: 128 for field in variant.inputs if field.default is _MISSING}
    for m, n, k in ((128, 128, 128), (128, 1, 128), (1, 128, 128), (16, 128, 4096), (32, 64, 4096)):
        values = {**required, "M": m, "N": n, "K": k, "n_elements": 2048, "n_cols": 64}
        if variant.matches(values):
            return values
    raise AssertionError(f"no integration input covers {variant.op_id}/{variant.name}")


def _expected_models(platform_key, model_version):
    models = {
        f"{platform_key}/flaggems/mul/broadcast_2d/bf16-bf16-bf16": {
            "path": "flaggems/mul/broadcast_2d/bf16-bf16-bf16/model.tar.gz",
        },
        f"{platform_key}/flaggems/mul/scalar/bf16-bf16": {
            "path": "flaggems/mul/scalar/bf16-bf16/model.tar.gz",
        },
    }
    if platform_key == "nvidia-h20":
        models.update({
            "nvidia-h20/flaggems/mm/gemv/bf16-bf16-bf16": {
                "path": "flaggems/mm/gemv/bf16-bf16-bf16/model.tar.gz",
            },
            "nvidia-h20/flaggems/mm/general_tma/bf16-bf16-bf16": {
                "path": "flaggems/mm/general_tma/bf16-bf16-bf16/model.tar.gz",
            },
            "nvidia-h20/flaggems/mm/splitk/bf16-bf16-bf16": {
                "path": "flaggems/mm/splitk/bf16-bf16-bf16/model.tar.gz",
            },
        })
    if platform_key == "thead-zw810e" and model_version == THEAD_MM_VERSION:
        for variant in (
                "gemv_ppu",
                "mm_ppu",
                "mm_ppu_mid_m",
                "mm_ppu_multi_row_gemv",
                "mm_ppu_narrow_columns",
                "mm_ppu_narrow_n",
                "mm_ppu_small_m",
                "mm_ppu_split_k",
        ):
            models[f"thead-zw810e/flaggems/mm/{variant}/bf16-bf16-bf16"] = {
                "path": f"flaggems/mm/{variant}/bf16-bf16-bf16/model.tar.gz",
            }
    return models


@pytest.mark.parametrize(("platform_key", "filename", "package_sha256"), PUBLISHED_PACKAGES)
def test_ksyun_package_download_validation_and_cache_reuse(
    tmp_path,
    monkeypatch,
    platform_key,
    filename,
    package_sha256,
):
    model_version = THEAD_MM_VERSION if filename.endswith("v1.1.0.tar.gz") else MODEL_VERSION
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
    monkeypatch.setenv("FLAGTUNE_MANIFEST_URL", REMOTE_MANIFEST_URL)
    staging_manifest = os.environ.get("FLAGTUNE_TEST_LOCAL_MANIFEST")
    if staging_manifest:
        monkeypatch.setenv("FLAGTUNE_LOCAL_MANIFEST", staging_manifest)

    pytest.importorskip("xgboost", reason="published model validation requires the real XGBoost runtime")
    downloaded = FlagTuneModelManager().resolve(
        "flaggems/mul",
        "broadcast_2d",
        platform_key=platform_key,
        dtype_key="bf16-bf16-bf16",
        model_version=model_version,
    )
    expected = (cache_root / "packages" / platform_key / model_version / f"{platform_key}_v{model_version}.tar.gz")
    remote_url = f"{REMOTE_BASE_URL}/{filename}"

    assert downloaded == expected
    assert downloaded.is_file()
    assert _sha256(downloaded) == package_sha256

    manifest_path = Path(staging_manifest) if staging_manifest else cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["packages"][platform_key]["versions"][model_version] == {
        "url": remote_url,
        "sha256": package_sha256,
    }

    parsed = read_platform_package(
        downloaded,
        expected_platform_key=platform_key,
        expected_version=model_version,
    )
    assert parsed.platform_key == platform_key
    assert parsed.package_version == model_version
    assert parsed.models == _expected_models(platform_key, model_version)

    monkeypatch.setenv("FLAGTUNE_DISABLE_REMOTE", "1")
    reused = FlagTuneModelManager().resolve(
        "flaggems/mul",
        "broadcast_2d",
        platform_key=platform_key,
        dtype_key="bf16-bf16-bf16",
        model_version=model_version,
    )
    assert reused == downloaded
    assert not list(cache_root.rglob("*.tmp"))

    # Loading through a new manager while offline exercises the disk cache,
    # real Booster/config binding and feature validation for every child.
    manager = FlagTuneModelManager()
    for artifact in sorted(parsed.models):
        platform, namespace, operator, variant_name, dtype_key = artifact.split("/")
        op_id = f"{namespace}/{operator}"
        loaded = manager.load(op_id, variant_name, platform_key=platform, dtype_key=dtype_key,
                              model_version=model_version)
        variant = loaded.variant
        values = _prediction_inputs(variant)
        candidates = list(islice(variant.iter_configs(), 128))
        assert candidates
        propose = make_config_proposer(op_id, variant_name, platform_key=platform, dtype_key=dtype_key,
                                       model_version=model_version)
        predicted = propose(None, values, candidates, {})
        assert predicted, artifact
        for candidate in predicted:
            assert set(candidate) == set(variant.param_names)
            assert variant.param_space.validate(candidate)
            assert candidate in candidates
