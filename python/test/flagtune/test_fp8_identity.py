"""Keep FP8 encodings distinct through training, packaging, and runtime loading."""

from __future__ import annotations

import json
from itertools import product

import numpy as np
import pytest

from triton.flagtune.contract.archive import (
    ModelArchiveError,
    platform_package_name,
    read_model_archive,
    read_model_archive_bytes,
    read_platform_package,
    write_platform_package,
)
from triton.flagtune.contract.identity import (
    ModelIdentity,
    ModelIdentityError,
    dtype_abbreviation,
    gpu_metadata,
    make_dtype_key,
    normalize_dtype_name,
)
from triton.flagtune.contract.operator_schema import (
    FlagTuneConfigError,
    model_identity_from_config,
    parse_model_config,
    parse_operator_config,
    variant_to_model_config,
)
from triton.flagtune.runtime.model_loader import FlagTuneModelManager, IncompatibleModelError
from triton.flagtune.training.ranker import (
    TrainingDataError,
    XGBoostTrainingOptions,
    export_ranker_model,
    prepare_ranking_data,
    train_xgboost_ranker,
)

FP8_FORMATS = (
    ("float8_e4m3fn", "f8e4m3fn", ("float8e4nv", "fp8e4nv")),
    ("float8_e4m3fnuz", "f8e4m3fnuz", ("float8e4b8", "fp8e4b8")),
    ("float8_e5m2", "f8e5m2", ("float8e5", "fp8e5")),
    ("float8_e5m2fnuz", "f8e5m2fnuz", ("float8e5b16", "fp8e5b16")),
)
GPU = dict(gpu_metadata(backend="cuda", vendor="nvidia", device_name="NVIDIA H20", architecture="sm90"))


def _variant():
    return parse_operator_config({
        "op_id": "tests/fp8",
        "variants": {
            "kernel": {
                "inputs": {"M": {}},
                "params": {"BLOCK": {"values": [16, 32, 64, 128]}},
                "features": ["M", "BLOCK"],
            },
        },
    }).get_variant("kernel")


def _identity(dtype):
    return ModelIdentity(GPU["platform_key"], "tests/fp8", "kernel", make_dtype_key([dtype, dtype, "bfloat16"]))


def _records(dtype):
    identity = _identity(dtype)
    return [{
        "schema_version": 2,
        "model_identity": {"platform_key": identity.platform_key, "dtype_key": identity.dtype_key},
        "dtypes": {"inputs": [dtype, dtype], "outputs": ["bfloat16"]},
        "device": {"metadata": GPU},
        "ranking_group": {
            "operator_id": identity.op_id,
            "variant": identity.variant,
            "dimensions": {"M": size},
            "model_dtype_key": identity.dtype_key,
        },
        "inputs": {"M": size},
        "config": {"BLOCK": block},
        "latency_ms": latency,
    }
            for size, latencies in ((64, (4.0, 3.0, 2.0, 1.0)), (128, (1.0, 2.0, 3.0, 4.0)))
            for block, latency in zip((16, 32, 64, 128), latencies)]


def _write_records(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


@pytest.mark.parametrize("canonical,short,aliases", FP8_FORMATS)
def test_fp8_dtype_names_round_trip(canonical, short, aliases):
    for name in (canonical, f"torch.{canonical}", short, *aliases):
        assert normalize_dtype_name(name) == canonical
        assert dtype_abbreviation(name) == short
        assert normalize_dtype_name(dtype_abbreviation(name)) == canonical
        assert make_dtype_key([name, "bfloat16"]) == f"{short}-bf16"
        assert make_dtype_key([name, name, "float32", "float32", "float16"]) == f"{short}-{short}-f32-f32-f16"


@pytest.mark.parametrize("canonical,short,_aliases", FP8_FORMATS)
def test_torch_fp8_dtype_objects_round_trip(canonical, short, _aliases):
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, canonical, None)
    if dtype is None:
        pytest.skip(f"installed torch does not expose {canonical}")
    assert normalize_dtype_name(dtype) == canonical
    assert dtype_abbreviation(dtype) == short


@pytest.mark.parametrize("dtype", ["fp8", "float8", "e4m3", "e5m2", "fp8e4b15", "float8e4b15"])
def test_ambiguous_or_distinct_unsupported_fp8_names_are_rejected(dtype):
    with pytest.raises(ModelIdentityError, match="unsupported tensor dtype"):
        normalize_dtype_name(dtype)


def test_fp8_dtype_keys_preserve_format_order_and_output_type():
    canonical_names = [canonical for canonical, _short, _aliases in FP8_FORMATS]
    keys = {
        make_dtype_key([left, right, output])
        for left, right, output in product(canonical_names, canonical_names, ("float16", "bfloat16"))
    }
    assert len(keys) == 32
    assert keys.isdisjoint(
        {make_dtype_key([dtype, dtype, output])
         for dtype, output in product(("float16", "bfloat16"), repeat=2)})
    assert make_dtype_key([canonical_names[0], "bfloat16"]) != make_dtype_key(["bfloat16", canonical_names[0]])


@pytest.mark.parametrize("canonical,short,aliases", FP8_FORMATS)
def test_fp8_model_config_preserves_canonical_dtype_identity(canonical, short, aliases):
    identity = _identity(canonical)
    config = variant_to_model_config(_variant(), identity, [f"torch.{canonical}", aliases[0], "bf16"], GPU, "1.0.0")

    assert config["dtype_key"] == f"{short}-{short}-bf16"
    assert config["dtypes"] == [canonical, canonical, "bfloat16"]
    assert model_identity_from_config(config) == identity
    assert parse_model_config(config).feature_names == _variant().feature_names

    wrong_format = next(name for name, _short, _aliases in FP8_FORMATS if name != canonical)
    with pytest.raises(FlagTuneConfigError, match="dtype_key does not match dtypes"):
        parse_model_config({**config, "dtype_key": _identity(wrong_format).dtype_key})
    with pytest.raises(FlagTuneConfigError, match="dtype_key does not match ordered dtypes"):
        variant_to_model_config(_variant(), _identity(wrong_format), config["dtypes"], GPU, "1.0.0")


@pytest.mark.parametrize("canonical,_short,_aliases", FP8_FORMATS)
def test_fp8_training_rejects_a_row_with_the_wrong_format(canonical, _short, _aliases, tmp_path):
    records = _records(canonical)
    wrong_format = next(name for name, _short, _aliases in FP8_FORMATS if name != canonical)
    records[0]["dtypes"]["inputs"][0] = wrong_format
    data_path = tmp_path / "benchmark.jsonl"
    _write_records(data_path, records)

    with pytest.raises(TrainingDataError, match="inconsistent model_identity.dtype_key"):
        prepare_ranking_data(_variant(), data_path, XGBoostTrainingOptions(min_train_rows=2, show_progress=False))


def test_fp8_cpu_training_export_package_and_runtime_round_trip(tmp_path, monkeypatch):
    """Train real CPU rankers; target GPU metadata never triggers device probing."""
    pytest.importorskip("xgboost")
    yaml = pytest.importorskip("yaml")
    variant = _variant()
    options = XGBoostTrainingOptions(n_estimators=4, max_depth=2, min_train_rows=2, n_jobs=1, show_progress=False)
    archives = {}
    expected_predictions = {}
    matrix = np.array([[64, 16], [64, 128], [128, 16], [128, 128]], dtype=np.float32)

    for canonical, short, aliases in FP8_FORMATS:
        identity = _identity(canonical)
        data_path = tmp_path / f"{short}.jsonl"
        _write_records(data_path, _records(canonical))
        data = prepare_ranking_data(variant, data_path, options)
        assert data.group_sizes == [4, 4]
        assert data.labels.tolist() == [0, 1, 2, 3, 3, 2, 1, 0]
        model, summary = train_xgboost_ranker(variant, data_path, options)
        exported = export_ranker_model(
            model,
            variant,
            tmp_path / "children",
            summary,
            identity=identity,
            dtypes=[f"torch.{canonical}", aliases[0], "bf16"],
            gpu=GPU,
            model_version="1.0.0",
        )
        assert exported.model_path == tmp_path / "children" / "tests" / "fp8" / "kernel" / identity.dtype_key / "model.tar.gz"
        members = read_model_archive(exported.model_path)
        config = yaml.safe_load(members["flagtune_config.yaml"])
        assert config["dtypes"] == [canonical, canonical, "bfloat16"]
        assert model_identity_from_config(config) == identity
        archives[identity] = exported.model_path
        expected_predictions[identity] = model.predict(matrix)

    models_root = tmp_path / "models"
    package_path = write_platform_package(
        models_root / platform_package_name(GPU["platform_key"], "1.0.0"),
        platform_key=GPU["platform_key"],
        package_version="1.0.0",
        model_archives=archives,
        required_identities=tuple(archives),
    )
    package = read_platform_package(package_path, expected_platform_key=GPU["platform_key"], expected_version="1.0.0")
    assert set(package.models) == {identity.artifact_key for identity in archives}
    assert len({entry["path"] for entry in package.models.values()}) == 4
    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(models_root))
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FLAGTUNE_DISABLE_REMOTE", "1")
    manager = FlagTuneModelManager()
    for identity, archive_path in archives.items():
        loaded = manager.load(
            identity.op_id,
            identity.variant,
            platform_key=identity.platform_key,
            dtype_key=identity.dtype_key,
            model_version="1.0.0",
        )
        assert loaded.identity == identity
        assert loaded.package_path == package_path
        assert loaded.model_member == package.models[identity.artifact_key]["path"]
        np.testing.assert_allclose(loaded.predictor.predict(matrix), expected_predictions[identity], rtol=0, atol=0)

        wrong_identity = next(other for other in archives if other != identity)
        with pytest.raises(ModelArchiveError, match="child model identity mismatch"):
            write_platform_package(
                tmp_path / "wrong-format.tar.gz",
                platform_key=identity.platform_key,
                package_version="1.0.0",
                model_archives={wrong_identity: archive_path},
            )
        members = read_model_archive_bytes(package.archives[identity.artifact_key])
        with pytest.raises(IncompatibleModelError, match="model identity mismatch"):
            manager._validate_bundle_members(wrong_identity, "1.0.0", members, str(archive_path))
