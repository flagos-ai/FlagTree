"""Test kernel-independent FlagTune ranking preparation, progress, and export.

The suite registers a synthetic operator, writes small contiguous JSONL groups,
and verifies feature order, deterministic sampling, grouping errors, portable
text progress, and compatibility of exported models with ``XGBRanker``.
"""

from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from triton.flagtune.contract.archive import (
    ModelArchiveError,
    read_model_archive,
    read_model_archive_bytes,
    read_platform_package,
    write_model_archive,
)
from triton.flagtune.contract.operator_schema import model_config_sha256, parse_operator_config
from triton.flagtune.contract.identity import ModelIdentity, gpu_metadata
from triton.flagtune.training.migration import migrate_model_archive, migrate_platform_package
from triton.flagtune.training.ranker import (
    TrainingDataError,
    XGBoostTrainingOptions,
    export_ranker_model,
    prepare_ranking_data,
    train_xgboost_ranker,
)
import triton.flagtune.training.ranker as training
import triton.flagtune.training.migration as migration

GPU = dict(gpu_metadata(
    backend="cuda",
    vendor="nvidia",
    device_name="NVIDIA H800 80GB HBM3",
    architecture="sm90",
))
IDENTITY = ModelIdentity(GPU["platform_key"], "tests/train", "kernel", "bf16-bf16-f32")
DTYPES = ["bfloat16", "bfloat16", "float32"]
MIGRATION_FEATURE_NAMES = ["M", "N", "BLOCK", "num_warps", "tile"]
MIGRATION_MATRIX = np.array([
    [64, 32, 16, 2, 32],
    [128, 64, 32, 4, 128],
    [96, 48, 32, 2, 64],
], dtype=np.float32)


def _config():
    """Return a minimal registry definition with raw and derived features."""
    return {
        "op_id": "tests/train",
        "variants": {
            "kernel": {
                "inputs": {"M": {}, "N": {}},
                "params": {
                    "BLOCK": {"values": [16, 32]},
                    "num_warps": {"values": [2, 4]},
                },
                "features": [
                    "M",
                    "N",
                    "BLOCK",
                    "num_warps",
                    {"name": "tile", "op": "mul", "args": ["BLOCK", "num_warps"]},
                ],
            }
        },
    }


def _write_data(path, shape_order=("a", "b")):
    """Write deterministic per-config latencies in requested shape-group order."""
    configs = [
        {"BLOCK": 16, "num_warps": 2},
        {"BLOCK": 16, "num_warps": 4},
        {"BLOCK": 32, "num_warps": 2},
        {"BLOCK": 32, "num_warps": 4},
    ]
    inputs = {"a": {"M": 64, "N": 32}, "b": {"M": 128, "N": 64}}
    latencies = {"a": [4.0, 3.0, 2.0, 1.0], "b": [1.0, 2.0, 3.0, 4.0]}
    with path.open("w", encoding="utf-8") as handle:
        for shape_name in shape_order:
            for config, latency in zip(configs, latencies[shape_name]):
                handle.write(
                    json.dumps({
                        "schema_version": 2,
                        "model_identity": {
                            "platform_key": GPU["platform_key"],
                            "dtype_key": IDENTITY.dtype_key,
                        },
                        "dtypes": {
                            "inputs": DTYPES[:2],
                            "outputs": DTYPES[2:],
                        },
                        "device": {
                            "metadata": GPU,
                        },
                        "ranking_group": {
                            "operator_id": "tests/train",
                            "variant": "kernel",
                            "dimensions": inputs[shape_name],
                            "model_dtype_key": "bf16-bf16-f32",
                        },
                        "inputs": inputs[shape_name],
                        "config": config,
                        "latency_ms": latency,
                    }) + "\n")


def _old_model_config_yaml(
    variant="kernel",
    op_id="tests/train",
    dtype_key="bf16-bf16-f32",
    dtypes=DTYPES,
):
    """Return the exact legacy model contract consumed only by migration."""
    dtype_lines = "".join(f"- {dtype}\n" for dtype in dtypes)
    return f"""format_version: 5
model_version: 0.9.0
flagtune_version_min: 0.2.0
gpu_key: nvidia-h20-sm90
op_id: {op_id}
variant: {variant}
dtype_key: {dtype_key}
dtypes:
{dtype_lines}gpu:
  backend: cuda
  vendor: NVIDIA
  device_name: NVIDIA H20-3e
  architecture: sm90
  gpu_key: nvidia-h20-sm90
inputs:
  M: {{}}
  N: {{}}
when: true
params:
  BLOCK:
    values:
    - 16
    - 32
  num_warps:
    values:
    - 2
    - 4
features:
- M
- N
- BLOCK
- num_warps
- name: tile
  op: mul
  args:
  - BLOCK
  - num_warps
""".encode("utf-8")


def _old_model_archive(
    tmp_path,
    variant="kernel",
    op_id="tests/train",
    dtype_key="bf16-bf16-f32",
    dtypes=DTYPES,
):
    """Fit one small legacy model before migration installs its no-fit guard."""
    xgboost = pytest.importorskip("xgboost")
    train_features = np.array([
        [64, 32, 16, 2, 32],
        [64, 32, 16, 4, 64],
        [64, 32, 32, 2, 64],
        [64, 32, 32, 4, 128],
    ], dtype=np.float32)
    labels = np.array([0, 1, 2, 3], dtype=np.float32)
    model = xgboost.XGBRanker(
        n_estimators=4,
        max_depth=2,
        n_jobs=1,
        objective="rank:pairwise",
        random_state=7,
    )
    model.fit(train_features, labels, group=[4], verbose=False)
    model.get_booster().feature_names = MIGRATION_FEATURE_NAMES
    yaml = pytest.importorskip("yaml")
    config_yaml = _old_model_config_yaml(variant, op_id, dtype_key, dtypes)
    old_config = yaml.safe_load(config_yaml)
    model.get_booster().set_attr(flagtune_config_sha256=model_config_sha256(old_config))
    model_path = tmp_path / f"{variant}.json"
    model.save_model(str(model_path))
    summary = {
        "feature_cols": MIGRATION_FEATURE_NAMES,
        "gpu_key": "nvidia-h20-sm90",
        "model_config_sha256": model_config_sha256(old_config),
        "model_identity": {
            "gpu_key": "nvidia-h20-sm90",
            "dtype_key": dtype_key,
        },
        "model_version": "0.9.0",
        "variant": variant,
    }
    archive_path = tmp_path / f"{variant}.tar.gz"
    write_model_archive(
        archive_path, {
            "xgboost_ranker.json": model_path.read_bytes(),
            "flagtune_config.yaml": config_yaml,
            "training_summary.json": json.dumps(summary, sort_keys=True).encode("utf-8"),
        })
    return archive_path, model_path.read_bytes(), old_config, model.predict(MIGRATION_MATRIX)


def test_prepare_ranking_data_preserves_feature_order_and_shape_groups(tmp_path):
    """Build float32 features, descending relevance labels, and group sizes."""
    variant = parse_operator_config(_config()).get_variant("kernel")
    data_path = tmp_path / "benchmark.jsonl"
    _write_data(data_path)

    data = prepare_ranking_data(
        variant,
        data_path,
        XGBoostTrainingOptions(min_train_rows=2, show_progress=False),
    )

    assert data.features.dtype == np.float32
    assert data.features.shape == (8, 5)
    assert data.group_sizes == [4, 4]
    assert data.labels.tolist() == [0, 1, 2, 3, 3, 2, 1, 0]
    assert data.features[0].tolist() == [64, 32, 16, 2, 32]


def test_prepare_ranking_data_sampling_is_reproducible(tmp_path):
    """Sample the same source rows for repeated calls with one fixed seed."""
    variant = parse_operator_config(_config()).get_variant("kernel")
    data_path = tmp_path / "benchmark.jsonl"
    _write_data(data_path)
    options = XGBoostTrainingOptions(
        min_train_rows=2,
        max_configs_per_shape=2,
        seed=17,
        show_progress=False,
    )

    first = prepare_ranking_data(variant, data_path, options)
    second = prepare_ranking_data(variant, data_path, options)

    assert first.group_sizes == [2, 2]
    assert first.sampled_out_rows == 4
    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.labels, second.labels)


def test_prepare_ranking_data_rejects_noncontiguous_shape_groups(tmp_path):
    """Reject a ranking group that reappears after a different query group."""
    variant = parse_operator_config(_config()).get_variant("kernel")
    data_path = tmp_path / "benchmark.jsonl"
    _write_data(data_path, shape_order=("a", "b", "a"))

    with pytest.raises(TrainingDataError, match="not contiguous"):
        prepare_ranking_data(
            variant,
            data_path,
            XGBoostTrainingOptions(min_train_rows=2, show_progress=False),
        )


def test_prepare_ranking_data_rejects_group_dimensions_mismatching_inputs(tmp_path):
    """Require the public ranking-group identity to match feature inputs."""
    variant = parse_operator_config(_config()).get_variant("kernel")
    data_path = tmp_path / "benchmark.jsonl"
    _write_data(data_path)
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    for row in rows[:4]:
        row["ranking_group"]["dimensions"]["M"] = 999
    data_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(TrainingDataError, match="dimensions do not match inputs"):
        prepare_ranking_data(
            variant,
            data_path,
            XGBoostTrainingOptions(min_train_rows=2, show_progress=False),
        )


def test_prepare_ranking_data_rejects_mixed_platform_identity(tmp_path):
    variant = parse_operator_config(_config()).get_variant("kernel")
    data_path = tmp_path / "benchmark.jsonl"
    _write_data(data_path)
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    for index, row in enumerate(rows):
        row.update({
            "model_identity": {
                "platform_key": ("nvidia-h800" if index < 4 else "nvidia-h20"),
                "dtype_key": "bf16-bf16-f32",
            },
            "dtypes": {
                "inputs": ["bfloat16", "bfloat16"],
                "outputs": ["float32"],
            },
        })
    data_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(TrainingDataError, match="mixes platform identities"):
        prepare_ranking_data(
            variant,
            data_path,
            XGBoostTrainingOptions(min_train_rows=2, show_progress=False),
        )


def test_prepare_ranking_data_rejects_mixed_device_architecture(tmp_path):
    """Keep one training corpus tied to one independently recorded target."""
    variant = parse_operator_config(_config()).get_variant("kernel")
    data_path = tmp_path / "benchmark.jsonl"
    _write_data(data_path)
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    rows[1]["device"]["metadata"]["architecture"] = "sm100"
    data_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(TrainingDataError, match="mixes device architectures"):
        prepare_ranking_data(
            variant,
            data_path,
            XGBoostTrainingOptions(min_train_rows=2, show_progress=False),
        )


def test_train_and_export_model_is_loadable_by_xgboost_ranker(tmp_path):
    """Fit a small ranker and reload its model/schema bundle with XGBoost."""
    xgboost = pytest.importorskip("xgboost")
    variant = parse_operator_config(_config()).get_variant("kernel")
    data_path = tmp_path / "benchmark.jsonl"
    _write_data(data_path)

    model, summary = train_xgboost_ranker(
        variant,
        data_path,
        XGBoostTrainingOptions(
            n_estimators=4,
            max_depth=2,
            min_train_rows=2,
            n_jobs=1,
            show_progress=False,
        ),
    )
    exported = export_ranker_model(
        model,
        variant,
        tmp_path,
        summary,
        identity=IDENTITY,
        dtypes=DTYPES,
        gpu=GPU,
        model_version="1.0.0",
    )

    loaded = xgboost.XGBRanker()
    members = read_model_archive(exported.model_path)
    loaded.load_model(bytearray(members["xgboost_ranker.json"]))
    assert len(loaded.predict(np.zeros((2, len(variant.feature_names))))) == 2
    yaml = pytest.importorskip("yaml")
    saved_config = yaml.safe_load(members["flagtune_config.yaml"])
    assert saved_config == exported.model_config
    assert saved_config["format_version"] == 5
    assert saved_config["model_version"] == "1.0.0"
    assert saved_config["platform_key"] == GPU["platform_key"]
    assert saved_config["dtype_key"] == "bf16-bf16-f32"
    assert saved_config["op_id"] == "tests/train"
    assert saved_config["variant"] == "kernel"
    assert exported.model_path == tmp_path / "tests" / "train" / "kernel" / "bf16-bf16-f32" / "model.tar.gz"
    assert list(exported.model_path.parent.iterdir()) == [exported.model_path]


def test_migrate_model_archive_preserves_predictions_without_fitting(tmp_path, monkeypatch):
    """Rewrite only legacy metadata while retaining the fitted Booster behavior."""
    xgboost = pytest.importorskip("xgboost")
    archive_path, _model_json, old_config, old_predictions = _old_model_archive(tmp_path)

    def reject_training(*_args, **_kwargs):
        raise AssertionError("migration must not train or fit an XGBoost model")

    monkeypatch.setattr(xgboost.XGBRanker, "fit", reject_training)
    monkeypatch.setattr(training, "train_xgboost_ranker", reject_training)
    identity, migrated_payload = migrate_model_archive(
        archive_path.read_bytes(),
        platform_key="nvidia-h20",
        model_version="1.0.0",
    )

    members = read_model_archive_bytes(migrated_payload, source="migrated test model")
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load(members["flagtune_config.yaml"])
    summary = json.loads(members["training_summary.json"])
    assert identity == ModelIdentity("nvidia-h20", "tests/train", "kernel", "bf16-bf16-f32")
    assert b"gpu_key" not in members["flagtune_config.yaml"]
    assert b"gpu_key" not in members["training_summary.json"]
    assert config["format_version"] == 5
    assert config["model_version"] == "1.0.0"
    assert config["platform_key"] == "nvidia-h20"
    assert config["gpu"]["platform_key"] == "nvidia-h20"
    for field in ("features", "params", "variant", "dtypes"):
        assert config[field] == old_config[field]

    digest = model_config_sha256(config)
    assert summary["model_config_sha256"] == digest
    assert summary["model_version"] == "1.0.0"
    migrated = xgboost.XGBRanker()
    migrated.load_model(bytearray(members["xgboost_ranker.json"]))
    assert migrated.get_booster().attr("flagtune_config_sha256") == digest
    assert migrated.get_booster().feature_names == MIGRATION_FEATURE_NAMES
    np.testing.assert_allclose(migrated.predict(MIGRATION_MATRIX), old_predictions, rtol=0, atol=0)


def test_migrate_model_archive_rejects_model_json_without_yaml(tmp_path):
    """Do not guess identity or schema from standalone XGBoost bytes."""
    _archive_path, model_json, _old_config, _old_predictions = _old_model_archive(tmp_path)

    with pytest.raises(ModelArchiveError, match="model archive"):
        migrate_model_archive(model_json, platform_key="nvidia-h20", model_version="1.0.0")


def test_migrate_model_archive_rejects_config_tampered_after_training(tmp_path):
    """Do not bless a legacy model paired with a different parameter contract."""
    archive_path, _model_json, _old_config, _old_predictions = _old_model_archive(tmp_path)
    members = read_model_archive(archive_path)
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load(members["flagtune_config.yaml"])
    config["params"]["BLOCK"]["values"] = [16, 64]
    members["flagtune_config.yaml"] = yaml.safe_dump(config, sort_keys=False).encode("utf-8")
    write_model_archive(archive_path, members)

    with pytest.raises(ModelArchiveError, match="legacy model config digest mismatch"):
        migrate_model_archive(archive_path.read_bytes(), platform_key="nvidia-h20", model_version="1.0.0")


def test_migrate_model_archive_rejects_legacy_gpu_key_mismatching_h20_metadata(tmp_path, ):
    """Do not relabel an archive whose old identity contradicts its H20 metadata."""
    xgboost = pytest.importorskip("xgboost")
    yaml = pytest.importorskip("yaml")
    archive_path, _model_json, _old_config, _old_predictions = _old_model_archive(tmp_path)
    members = read_model_archive(archive_path)
    config = yaml.safe_load(members["flagtune_config.yaml"])
    config["gpu_key"] = "amd-mi300x-gfx942"
    config["gpu"]["gpu_key"] = "amd-mi300x-gfx942"
    members["flagtune_config.yaml"] = yaml.safe_dump(config, sort_keys=False).encode("utf-8")
    model = xgboost.XGBRanker()
    model.load_model(bytearray(members["xgboost_ranker.json"]))
    model.get_booster().set_attr(flagtune_config_sha256=model_config_sha256(config))
    model_path = tmp_path / "mismatched-identity.json"
    model.save_model(str(model_path))
    members["xgboost_ranker.json"] = model_path.read_bytes()
    write_model_archive(archive_path, members)

    with pytest.raises(ModelArchiveError, match="legacy gpu_key"):
        migrate_model_archive(
            archive_path.read_bytes(),
            platform_key="nvidia-h20",
            model_version="1.0.0",
        )


def test_migrate_model_archive_requires_sm90_architecture(tmp_path):
    """H20 migration accepts legacy identity only for the published architecture."""
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load(_old_model_config_yaml())
    config["gpu"]["architecture"] = "sm89"
    config["gpu_key"] = "nvidia-h20-sm89"
    config["gpu"]["gpu_key"] = "nvidia-h20-sm89"

    with pytest.raises(ModelArchiveError, match="architecture.*sm90"):
        migration._migrate_config(config, "nvidia-h20", "1.0.0")


def test_migrate_model_archive_rejects_booster_feature_order_mismatch(tmp_path):
    """Reject a Booster that cannot satisfy the migrated config feature contract."""
    xgboost = pytest.importorskip("xgboost")
    archive_path, _model_json, _old_config, _old_predictions = _old_model_archive(tmp_path)
    members = read_model_archive(archive_path)
    model = xgboost.XGBRanker()
    model.load_model(bytearray(members["xgboost_ranker.json"]))
    model.get_booster().feature_names = list(reversed(MIGRATION_FEATURE_NAMES))
    model_path = tmp_path / "reordered-features.json"
    model.save_model(str(model_path))
    members["xgboost_ranker.json"] = model_path.read_bytes()
    write_model_archive(archive_path, members)

    with pytest.raises(ModelArchiveError, match="feature order"):
        migrate_model_archive(
            archive_path.read_bytes(),
            platform_key="nvidia-h20",
            model_version="1.0.0",
        )


def test_migrate_model_archive_rejects_summary_feature_contract_mismatch(tmp_path):
    """Keep the migrated audit summary bound to the same ordered feature schema."""
    archive_path, _model_json, _old_config, _old_predictions = _old_model_archive(tmp_path)
    members = read_model_archive(archive_path)
    summary = json.loads(members["training_summary.json"])
    summary["feature_cols"] = list(reversed(MIGRATION_FEATURE_NAMES))
    members["training_summary.json"] = json.dumps(summary, sort_keys=True).encode("utf-8")
    write_model_archive(archive_path, members)

    with pytest.raises(ModelArchiveError, match="summary feature_cols"):
        migrate_model_archive(
            archive_path.read_bytes(),
            platform_key="nvidia-h20",
            model_version="1.0.0",
        )


def test_migrate_platform_package_writes_all_required_models(tmp_path):
    variants = ("gemv", "general_tma", "splitk")
    archives = [_old_model_archive(tmp_path, variant)[0] for variant in variants]
    required = tuple(ModelIdentity("nvidia-h20", "tests/train", variant, "bf16-bf16-f32") for variant in variants)
    output = tmp_path / "nvidia-h20_v1.0.0.tar.gz"

    result = migrate_platform_package(
        archives,
        platform_key="nvidia-h20",
        model_version="1.0.0",
        output_path=output,
        required_identities=required,
    )

    package = read_platform_package(result, expected_platform_key="nvidia-h20", expected_version="1.0.0")
    assert result == output
    assert set(package.models) == {identity.artifact_key for identity in required}


def test_migrate_platform_package_rejects_missing_required_model(tmp_path):
    variants = ("gemv", "general_tma", "splitk")
    archives = [_old_model_archive(tmp_path, variant)[0] for variant in variants[:2]]
    required = tuple(ModelIdentity("nvidia-h20", "tests/train", variant, "bf16-bf16-f32") for variant in variants)
    output = tmp_path / "nvidia-h20_v1.0.0.tar.gz"

    with pytest.raises(ModelArchiveError, match="missing required identities"):
        migrate_platform_package(
            archives,
            platform_key="nvidia-h20",
            model_version="1.0.0",
            output_path=output,
            required_identities=required,
        )
    assert not output.exists()


def test_migrate_platform_package_rejects_noncanonical_output_name(tmp_path):
    variants = ("gemv", "general_tma", "splitk")
    archives = [_old_model_archive(tmp_path, variant)[0] for variant in variants]
    required = tuple(ModelIdentity("nvidia-h20", "tests/train", variant, "bf16-bf16-f32") for variant in variants)
    output = tmp_path / "wrong-name.tar.gz"

    with pytest.raises(ModelArchiveError, match="package filename"):
        migrate_platform_package(
            archives,
            platform_key="nvidia-h20",
            model_version="1.0.0",
            output_path=output,
            required_identities=required,
        )
    assert not output.exists()


@pytest.mark.parametrize("manifest_state", ["disabled", "absent", "empty", "placeholder"])
def test_migration_cli_writes_complete_h20_package(tmp_path, capsys, manifest_state):
    variants = ("gemv", "general_tma", "splitk")
    archives = [
        _old_model_archive(
            tmp_path,
            variant,
            op_id="flaggems/mm",
            dtype_key="bf16-bf16-bf16",
            dtypes=("bfloat16", "bfloat16", "bfloat16"),
        )[0] for variant in variants
    ]
    output = tmp_path / "nvidia-h20_v1.0.0.tar.gz"
    manifest_output = tmp_path / "flagtune-manifest.json"
    package_url = "https://models.example.com/flagtune/flagtune-xgb-nvidia-h20_v0.1.0.tar.gz"
    arguments = [
        "--platform-key",
        "nvidia-h20",
        "--model-version",
        "1.0.0",
        "--output",
        str(output),
    ]
    publish_manifest = manifest_state != "disabled"
    if manifest_state == "empty":
        manifest_output.touch()
    elif manifest_state == "placeholder":
        manifest_output.write_text(
            '{"schema_version": 1, "packages": {}}\n',
            encoding="utf-8",
        )
    if publish_manifest:
        arguments.extend((
            "--manifest-output",
            str(manifest_output),
            "--package-url",
            package_url,
        ))
    for archive in archives:
        arguments.extend(("--model", str(archive)))

    assert migration.main(arguments) == 0

    package = read_platform_package(output, expected_platform_key="nvidia-h20", expected_version="1.0.0")
    required = {
        ModelIdentity("nvidia-h20", "flaggems/mm", variant, "bf16-bf16-bf16").artifact_key
        for variant in variants
    }
    assert set(package.models) == required
    if publish_manifest:
        assert json.loads(manifest_output.read_text(encoding="utf-8")) == {
            "schema_version": 1,
            "packages": {
                "nvidia-h20": {
                    "versions": {
                        "1.0.0": {
                            "url": package_url,
                            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                        },
                    },
                },
            },
        }
    else:
        assert not manifest_output.exists()
    assert capsys.readouterr().out.strip() == str(output)


@pytest.mark.parametrize(
    "publishing_argument",
    [
        ("--manifest-output", "flagtune-manifest.json"),
        ("--package-url", "https://models.example.com/flagtune/nvidia-h20_v1.0.0.tar.gz"),
    ],
)
def test_migration_cli_requires_manifest_arguments_together(
    tmp_path,
    publishing_argument,
    capsys,
):
    output = tmp_path / "nvidia-h20_v1.0.0.tar.gz"
    arguments = [
        "--platform-key",
        "nvidia-h20",
        "--model-version",
        "1.0.0",
        "--output",
        str(output),
        "--model",
        str(tmp_path / "does-not-exist.tar.gz"),
        publishing_argument[0],
        str(tmp_path / publishing_argument[1]),
    ]

    with pytest.raises(SystemExit) as exc_info:
        migration.main(arguments)

    assert exc_info.value.code == 2
    assert "--manifest-output and --package-url must be provided together" in capsys.readouterr().err
    assert not output.exists()


def test_migration_cli_rejects_non_https_package_url_before_migration(tmp_path):
    output = tmp_path / "nvidia-h20_v1.0.0.tar.gz"
    manifest_output = tmp_path / "flagtune-manifest.json"

    with pytest.raises(ModelArchiveError, match="package URL must use HTTPS"):
        migration.main([
            "--platform-key",
            "nvidia-h20",
            "--model-version",
            "1.0.0",
            "--output",
            str(output),
            "--model",
            str(tmp_path / "does-not-exist.tar.gz"),
            "--manifest-output",
            str(manifest_output),
            "--package-url",
            "http://models.example.com/flagtune/wrong-name.tar.gz",
        ])

    assert not output.exists()
    assert not manifest_output.exists()


def test_migration_cli_rejects_existing_nonempty_manifest_before_migration(tmp_path):
    output = tmp_path / "nvidia-h20_v1.0.0.tar.gz"
    manifest_output = tmp_path / "flagtune-manifest.json"
    manifest_output.write_text('{"existing": true}\n', encoding="utf-8")

    with pytest.raises(ModelArchiveError, match="Manifest output already exists and is not empty"):
        migration.main([
            "--platform-key",
            "nvidia-h20",
            "--model-version",
            "1.0.0",
            "--output",
            str(output),
            "--model",
            str(tmp_path / "does-not-exist.tar.gz"),
            "--manifest-output",
            str(manifest_output),
            "--package-url",
            "https://models.example.com/flagtune/nvidia-h20_v1.0.0.tar.gz",
        ])

    assert not output.exists()
    assert manifest_output.read_text(encoding="utf-8") == '{"existing": true}\n'


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_migration_cli_rejects_placeholder_with_noninteger_schema_version(
    tmp_path,
    schema_version,
):
    output = tmp_path / "nvidia-h20_v1.0.0.tar.gz"
    manifest_output = tmp_path / "flagtune-manifest.json"
    manifest_output.write_text(
        json.dumps({"schema_version": schema_version, "packages": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ModelArchiveError, match="Manifest output already exists and is not empty"):
        migration.main([
            "--platform-key",
            "nvidia-h20",
            "--model-version",
            "1.0.0",
            "--output",
            str(output),
            "--model",
            str(tmp_path / "does-not-exist.tar.gz"),
            "--manifest-output",
            str(manifest_output),
            "--package-url",
            "https://models.example.com/flagtune/nvidia-h20_v1.0.0.tar.gz",
        ])

    assert not output.exists()


def test_migration_cli_preserves_manifest_changed_during_migration(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "nvidia-h20_v1.0.0.tar.gz"
    manifest_output = tmp_path / "flagtune-manifest.json"
    manifest_output.write_text(
        '{"schema_version": 1, "packages": {}}\n',
        encoding="utf-8",
    )
    unrelated_manifest = '{"owned_by_another_process": true}\n'

    def migrate_while_manifest_changes(
        model_paths,
        *,
        platform_key,
        model_version,
        output_path,
        required_identities,
    ):
        del model_paths, platform_key, model_version, required_identities
        package_path = Path(output_path)
        package_path.write_bytes(b"completed outer package")
        manifest_output.write_text(unrelated_manifest, encoding="utf-8")
        return package_path

    monkeypatch.setattr(
        migration,
        "migrate_platform_package",
        migrate_while_manifest_changes,
    )

    with pytest.raises(ModelArchiveError, match="Manifest output changed during publishing"):
        migration.main([
            "--platform-key",
            "nvidia-h20",
            "--model-version",
            "1.0.0",
            "--output",
            str(output),
            "--model",
            str(tmp_path / "legacy-model.tar.gz"),
            "--manifest-output",
            str(manifest_output),
            "--package-url",
            "https://models.example.com/flagtune/nvidia-h20_v1.0.0.tar.gz",
        ])

    assert manifest_output.read_text(encoding="utf-8") == unrelated_manifest


def test_migration_cli_preserves_manifest_created_at_publication_boundary(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "nvidia-h20_v1.0.0.tar.gz"
    manifest_output = tmp_path / "flagtune-manifest.json"
    unrelated_manifest = b'{"owned_by_another_process": true}\n'
    real_link = migration.os.link

    def create_manifest_before_no_clobber_publication(source, destination):
        manifest_output.write_bytes(unrelated_manifest)
        real_link(source, destination)

    def migrate_without_models(
        model_paths,
        *,
        platform_key,
        model_version,
        output_path,
        required_identities,
    ):
        del model_paths, platform_key, model_version, required_identities
        package_path = Path(output_path)
        package_path.write_bytes(b"completed outer package")
        return package_path

    monkeypatch.setattr(migration.os, "link", create_manifest_before_no_clobber_publication)
    monkeypatch.setattr(migration, "migrate_platform_package", migrate_without_models)

    with pytest.raises(ModelArchiveError, match="Manifest output changed during publishing"):
        migration.main([
            "--platform-key",
            "nvidia-h20",
            "--model-version",
            "1.0.0",
            "--output",
            str(output),
            "--model",
            str(tmp_path / "legacy-model.tar.gz"),
            "--manifest-output",
            str(manifest_output),
            "--package-url",
            "https://models.example.com/flagtune/nvidia-h20_v1.0.0.tar.gz",
        ])

    assert manifest_output.read_bytes() == unrelated_manifest


def test_migration_cli_reuses_sidecar_lock_left_by_terminated_process(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "nvidia-h20_v1.0.0.tar.gz"
    manifest_output = tmp_path / "flagtune-manifest.json"
    lock_path = tmp_path / ".flagtune-manifest.json.lock"
    lock_path.write_text("pid=999999999\n", encoding="utf-8")

    def migrate_without_models(
        model_paths,
        *,
        platform_key,
        model_version,
        output_path,
        required_identities,
    ):
        del model_paths, platform_key, model_version, required_identities
        package_path = Path(output_path)
        package_path.write_bytes(b"completed outer package")
        return package_path

    monkeypatch.setattr(migration, "migrate_platform_package", migrate_without_models)

    assert migration.main([
        "--platform-key",
        "nvidia-h20",
        "--model-version",
        "1.0.0",
        "--output",
        str(output),
        "--model",
        str(tmp_path / "legacy-model.tar.gz"),
        "--manifest-output",
        str(manifest_output),
        "--package-url",
        "https://models.example.com/flagtune/nvidia-h20_v1.0.0.tar.gz",
    ]) == 0

    assert lock_path.is_file()
    assert json.loads(manifest_output.read_text(encoding="utf-8"))["schema_version"] == 1


def test_migration_cli_rejects_non_h20_platform_before_reading_models(tmp_path):
    with pytest.raises(ModelArchiveError, match="only supports platform 'nvidia-h20'"):
        migration.main([
            "--platform-key",
            "amd-mi300x",
            "--model-version",
            "1.0.0",
            "--output",
            str(tmp_path / "amd-mi300x_v1.0.0.tar.gz"),
            "--model",
            str(tmp_path / "does-not-exist.tar.gz"),
        ])


def test_xgboost_progress_uses_flushed_text_without_tqdm(monkeypatch, capsys):
    """Report boosting rounds through the console fallback when tqdm is absent."""
    real_import = builtins.__import__

    def import_without_tqdm(name, globals=None, locals=None, fromlist=(), level=0):
        """Raise only for tqdm imports and delegate every other import unchanged."""
        if name == "tqdm" or name.startswith("tqdm."):
            raise ImportError("tqdm intentionally unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_tqdm)
    callbacks, progress = training._progress_callback(total=4, enabled=True)

    assert progress is None
    assert len(callbacks) == 1
    for epoch in range(4):
        assert callbacks[0].after_iteration(None, epoch, {}) is False
    output = capsys.readouterr().out
    assert "XGBoost progress: 1/4 trees" in output
    assert "XGBoost progress: 4/4 trees" in output
