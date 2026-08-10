from __future__ import annotations

import math
import numpy as np
import pytest

from triton.flagtune.runtime import proposer as predict
from triton.flagtune.contract.archive import (
    platform_package_name,
    read_model_archive,
    write_model_archive,
    write_platform_package,
)
from triton.flagtune.contract.identity import (
    ModelIdentity,
    ModelIdentityError,
    artifact_key,
    gpu_metadata,
    make_dtype_key,
    make_platform_key,
    normalize_dtype_name,
)
from triton.flagtune.runtime.model_loader import FlagTuneModelManager, IncompatibleModelError
from triton.flagtune.contract.operator_schema import (
    BUILTIN_OPS,
    FlagTuneConfigError,
    load_model_config,
    load_operator_config,
    model_config_sha256,
    parse_model_config,
    parse_operator_config,
    variant_to_model_config,
)

GPU = dict(gpu_metadata(
    backend="cuda",
    vendor="nvidia",
    device_name="NVIDIA H800 80GB HBM3",
    architecture="sm90",
))
PLATFORM_KEY = GPU["platform_key"]
DTYPES = ["bfloat16", "bfloat16", "float32"]
DTYPE_KEY = "bf16-bf16-f32"
MODEL_VERSION = "1.2.3"


def _identity(op_id="vendor/mm", variant="general"):
    return ModelIdentity(PLATFORM_KEY, op_id, variant, DTYPE_KEY)


def _config():
    return {
        "op_id": "vendor/mm",
        "variants": {
            "general": {
                "inputs": {
                    "M": {"min": 1},
                    "N": {},
                    "K": {},
                    "stride_am": {"default": "K"},
                    "stride_bk": {"default": "N"},
                },
                "when": {"op": "gt", "args": ["N", 1]},
                "params": {
                    "BLOCK_M": {"values": [16, 32]},
                    "num_warps": {"values": [4, 8]},
                },
                "features": [
                    "M",
                    {"name": "N", "op": "ident", "args": ["N"]},
                    {"name": "tile", "op": "mul", "args": ["M", "BLOCK_M"]},
                    {"name": "grid", "op": "cdiv", "args": ["M", "BLOCK_M"]},
                    {"name": "ratio", "op": "fdiv", "args": ["BLOCK_M", "M"]},
                    {"name": "aligned", "op": "alignup", "args": ["M", 16]},
                    {"name": "power", "op": "pow", "args": [2, 3]},
                    {"name": "log_tile", "op": "log2", "args": ["tile"]},
                ],
            }
        },
    }


def _export_model_archive(root, model_version):
    xgboost = pytest.importorskip("xgboost")
    from triton.flagtune.training.ranker import export_ranker_model

    variant = parse_operator_config(_config()).get_variant("general")
    model = xgboost.XGBRanker(n_estimators=0)
    model.fit(np.zeros((2, len(variant.feature_names))), np.zeros(2), group=[2])
    return export_ranker_model(
        model,
        variant,
        root,
        {},
        identity=_identity(),
        dtypes=DTYPES,
        gpu=GPU,
        model_version=model_version,
    ).model_path


def _export_platform_package(root, model_version):
    child = _export_model_archive(root / f"child-{model_version}", model_version)
    package = root / platform_package_name(PLATFORM_KEY, model_version)
    write_platform_package(
        package,
        platform_key=PLATFORM_KEY,
        package_version=model_version,
        model_archives={_identity(): child},
    )
    return package, child


@pytest.fixture(autouse=True)
def clean_model_manager(monkeypatch):
    """Reset lazy runtime caches and environment overrides around each test."""
    predict._MODEL_MANAGER = None
    predict._TOP_K_CACHE = None
    monkeypatch.delenv("FLAGTUNE_DISABLE_OPS", raising=False)
    monkeypatch.delenv("FLAGTUNE_TOP_K", raising=False)
    yield
    predict._MODEL_MANAGER = None
    predict._TOP_K_CACHE = None


def test_parse_operator_builds_inputs_params_and_ordered_features():
    """Compile an operator without changing process-global state."""
    info = parse_operator_config(_config())
    variant = info.get_variant("general")
    inputs = variant.normalize_inputs({"M": 33, "N": 8, "K": 64})

    assert inputs == {"M": 33, "N": 8, "K": 64, "stride_am": 64, "stride_bk": 8}
    assert variant.matches(inputs)
    assert list(variant.iter_configs()) == [
        {"BLOCK_M": 16, "num_warps": 4},
        {"BLOCK_M": 16, "num_warps": 8},
        {"BLOCK_M": 32, "num_warps": 4},
        {"BLOCK_M": 32, "num_warps": 8},
    ]

    rows = variant.build_feature_rows(inputs, [{"BLOCK_M": 16, "num_warps": 4}])
    assert list(rows[0]) == ["M", "N", "tile", "grid", "ratio", "aligned", "power", "log_tile"]
    assert rows[0] == pytest.approx({
        "M": 33,
        "N": 8,
        "tile": 528,
        "grid": 3,
        "ratio": 16 / 33,
        "aligned": 48,
        "power": 8,
        "log_tile": math.log2(528),
    })


def test_when_rejects_wrong_variant_shape():
    variant = parse_operator_config(_config()).get_variant("general")
    assert not variant.matches({"M": 32, "N": 1, "K": 64})


@pytest.mark.parametrize(
    "disabled",
    ["*", "vendor/mm", "vendor/mm/general", f"{PLATFORM_KEY}/vendor/mm/general/{DTYPE_KEY}"],
)
def test_disable_rules_accept_global_operator_and_exact_pair(monkeypatch, disabled):
    """Disable without loading a bundle at all three supported scopes."""
    monkeypatch.setenv("FLAGTUNE_DISABLE_OPS", disabled)
    proposer = predict.make_config_proposer("vendor/mm", "general", platform_key=PLATFORM_KEY, dtype_key=DTYPE_KEY)
    assert proposer(None, {}, [], {}) == []


def test_legacy_identity_alias_is_rejected():
    """Keep the operator/variant pair as the only model identity."""
    config = _config()
    legacy_key = "model" + "_" + "id"
    config["variants"]["general"][legacy_key] = "legacy/general"
    with pytest.raises(FlagTuneConfigError, match="unknown keys"):
        parse_operator_config(config)


def test_builtin_comparison_logic_and_power_operations_are_available():
    required = {
        "ident",
        "add",
        "sub",
        "mul",
        "div",
        "cdiv",
        "fdiv",
        "mod",
        "log2",
        "pow",
        "alignup",
        "aligndown",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "all",
        "any",
        "not",
    }
    assert required <= set(BUILTIN_OPS)
    assert BUILTIN_OPS["pow"](2, 5) == 32


def test_gpu_and_ordered_tensor_dtype_identity_is_canonical():
    assert make_platform_key("NVIDIA", "NVIDIA H800 80GB HBM3") == "nvidia-h800-80gb-hbm3"
    assert make_platform_key("NVIDIA", "NVIDIA H800 80GB HBM3") == PLATFORM_KEY
    assert normalize_dtype_name("torch.bfloat16") == "bfloat16"
    assert make_dtype_key(["bfloat16", "float16", "float32"]) == "bf16-f16-f32"
    with pytest.raises(ModelIdentityError, match="unsupported tensor dtype"):
        make_dtype_key(["float8_unknown"])


def test_yaml_loading_forwards_to_stateless_compiler(tmp_path):
    pytest.importorskip("yaml")
    config_path = tmp_path / "operator.yaml"
    config_path.write_text(
        """
op_id: vendor/add
variants:
  general:
    inputs:
      N: {}
    params:
      BLOCK: {values: [32]}
    features:
      - N
      - {name: BLOCK, op: ident, args: [BLOCK]}
""".strip(),
        encoding="utf-8",
    )
    info = load_operator_config(config_path)
    assert info.op_id == "vendor/add"
    assert artifact_key(PLATFORM_KEY, info.op_id, "general",
                        DTYPE_KEY) == (f"{PLATFORM_KEY}/vendor/add/general/{DTYPE_KEY}")


def test_registration_rejects_unknown_variables_and_unsafe_identities():
    bad_feature = _config()
    bad_feature["variants"]["general"]["features"].append({"name": "bad", "op": "ident", "args": ["missing"]})
    with pytest.raises(FlagTuneConfigError, match="unknown symbol"):
        parse_operator_config(bad_feature)

    bad_op = _config()
    bad_op["op_id"] = "../outside"
    with pytest.raises(FlagTuneConfigError, match="segment"):
        parse_operator_config(bad_op)

    bad_variant = _config()
    bad_variant["variants"]["bad/name"] = bad_variant["variants"].pop("general")
    with pytest.raises(FlagTuneConfigError, match="variants key"):
        parse_operator_config(bad_variant)


def test_single_model_config_round_trip_preserves_contract(tmp_path):
    """Serialize and compile one variant with ordered inputs, params, and features."""
    yaml = pytest.importorskip("yaml")
    variant = parse_operator_config(_config()).get_variant("general")
    config = variant_to_model_config(variant, _identity(), DTYPES, GPU, MODEL_VERSION)
    path = tmp_path / "flagtune_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    loaded, raw = load_model_config(path)
    assert (loaded.op_id, loaded.name) == (variant.op_id, variant.name)
    assert loaded.feature_names == variant.feature_names
    assert loaded.param_names == variant.param_names
    assert model_config_sha256(raw) == model_config_sha256(config)
    other_version = variant_to_model_config(variant, _identity(), DTYPES, GPU, "1.2.4")
    assert model_config_sha256(other_version) != model_config_sha256(config)


def test_model_config_rejects_unknown_custom_operation():
    """Keep exported bundles independent from external Python callables."""
    config = variant_to_model_config(
        parse_operator_config(_config()).get_variant("general"),
        _identity(),
        DTYPES,
        GPU,
        MODEL_VERSION,
    )
    config["when"] = {"op": "external_policy", "args": ["inputs"]}
    with pytest.raises(FlagTuneConfigError, match="unknown operation"):
        parse_model_config(config)


def test_untrained_empty_xgboost_model_runs_the_candidate_pipeline(tmp_path, monkeypatch):
    xgboost = pytest.importorskip("xgboost")
    from triton.flagtune.training.ranker import export_ranker_model

    feature_names = ["M", "N", "tile", "grid", "ratio", "aligned", "power", "log_tile"]
    empty_model = xgboost.XGBRanker(n_estimators=0)
    empty_model.fit(np.zeros((2, len(feature_names))), np.zeros(2), group=[2])
    variant = parse_operator_config(_config()).get_variant("general")
    exported = export_ranker_model(
        empty_model,
        variant,
        tmp_path / "child",
        {},
        identity=_identity(),
        dtypes=DTYPES,
        gpu=GPU,
        model_version=MODEL_VERSION,
    )
    write_platform_package(
        tmp_path / platform_package_name(PLATFORM_KEY, MODEL_VERSION),
        platform_key=PLATFORM_KEY,
        package_version=MODEL_VERSION,
        model_archives={_identity(): exported.model_path},
    )
    assert empty_model.get_booster().get_dump() == []

    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("FLAGTUNE_TOP_K", "2")

    proposer = predict.make_config_proposer("vendor/mm", "general", platform_key=PLATFORM_KEY, dtype_key=DTYPE_KEY)
    result = proposer(None, {"M": 33, "N": 8, "K": 64}, [], {})
    assert result == [
        {"BLOCK_M": 16, "num_warps": 4},
        {"BLOCK_M": 16, "num_warps": 8},
    ]


def test_loaded_model_cache_isolated_by_explicit_version(tmp_path, monkeypatch):
    """Keep two revisions of the same four-component identity independent."""
    for version in ("1.0.0", "2.0.0"):
        _export_platform_package(tmp_path, version)
    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(tmp_path))
    manager = FlagTuneModelManager()

    first = manager.load("vendor/mm", "general", platform_key=PLATFORM_KEY, dtype_key=DTYPE_KEY, model_version="1.0.0")
    second = manager.load("vendor/mm", "general", platform_key=PLATFORM_KEY, dtype_key=DTYPE_KEY, model_version="2.0.0")
    latest = manager.load("vendor/mm", "general", platform_key=PLATFORM_KEY, dtype_key=DTYPE_KEY)
    assert first.model_version == "1.0.0"
    assert second.model_version == latest.model_version == "2.0.0"
    assert first.package_path != second.package_path
    assert second is latest


def test_implicit_load_reuses_first_bundle_before_reresolution(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    _export_platform_package(model_root, "1.0.0")
    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(model_root))
    manager = FlagTuneModelManager()
    resolve_calls = []
    real_resolve = manager.resolve

    def counted_resolve(*args, **kwargs):
        resolve_calls.append(kwargs.get("model_version"))
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(manager, "resolve", counted_resolve)
    first = manager.load("vendor/mm", "general", platform_key=PLATFORM_KEY, dtype_key=DTYPE_KEY)

    _export_platform_package(model_root, "2.0.0")
    monkeypatch.setenv("FLAGTUNE_MODEL_VERSION", "2.0.0")
    second = manager.load("vendor/mm", "general", platform_key=PLATFORM_KEY, dtype_key=DTYPE_KEY)

    assert second is first
    assert second.model_version == "1.0.0"
    assert resolve_calls == [None]


def test_modified_config_is_rejected_by_embedded_model_digest(tmp_path, monkeypatch):
    """Reject a valid YAML config that no longer belongs to its XGBoost file."""
    xgboost = pytest.importorskip("xgboost")
    yaml = pytest.importorskip("yaml")
    from triton.flagtune.training.ranker import export_ranker_model

    variant = parse_operator_config(_config()).get_variant("general")
    model = xgboost.XGBRanker(n_estimators=0)
    model.fit(np.zeros((2, len(variant.feature_names))), np.zeros(2), group=[2])
    exported = export_ranker_model(
        model,
        variant,
        tmp_path / "child",
        {},
        identity=_identity(),
        dtypes=DTYPES,
        gpu=GPU,
        model_version=MODEL_VERSION,
    )
    members = read_model_archive(exported.model_path)
    config = yaml.safe_load(members["flagtune_config.yaml"])
    config["features"].pop()
    members["flagtune_config.yaml"] = yaml.safe_dump(config, sort_keys=False).encode()
    write_model_archive(exported.model_path, members)
    write_platform_package(
        tmp_path / platform_package_name(PLATFORM_KEY, MODEL_VERSION),
        platform_key=PLATFORM_KEY,
        package_version=MODEL_VERSION,
        model_archives={_identity(): exported.model_path},
    )
    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(tmp_path))

    with pytest.raises(IncompatibleModelError, match="digest mismatch"):
        FlagTuneModelManager().load("vendor/mm", "general", platform_key=PLATFORM_KEY, dtype_key=DTYPE_KEY)


def test_embedded_xgboost_feature_order_must_match_config(tmp_path, monkeypatch):
    """Reject weights whose named columns were reordered after export."""
    xgboost = pytest.importorskip("xgboost")
    from triton.flagtune.training.ranker import export_ranker_model

    variant = parse_operator_config(_config()).get_variant("general")
    model = xgboost.XGBRanker(n_estimators=0)
    model.fit(np.zeros((2, len(variant.feature_names))), np.zeros(2), group=[2])
    exported = export_ranker_model(
        model,
        variant,
        tmp_path / "child",
        {},
        identity=_identity(),
        dtypes=DTYPES,
        gpu=GPU,
        model_version=MODEL_VERSION,
    )
    members = read_model_archive(exported.model_path)
    changed = xgboost.XGBRanker()
    changed.load_model(bytearray(members["xgboost_ranker.json"]))
    changed.get_booster().feature_names = list(reversed(variant.feature_names))
    loose_path = tmp_path / "changed.json"
    changed.save_model(str(loose_path))
    members["xgboost_ranker.json"] = loose_path.read_bytes()
    write_model_archive(exported.model_path, members)
    write_platform_package(
        tmp_path / platform_package_name(PLATFORM_KEY, MODEL_VERSION),
        platform_key=PLATFORM_KEY,
        package_version=MODEL_VERSION,
        model_archives={_identity(): exported.model_path},
    )
    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(tmp_path))

    with pytest.raises(IncompatibleModelError, match="feature order mismatch"):
        FlagTuneModelManager().load("vendor/mm", "general", platform_key=PLATFORM_KEY, dtype_key=DTYPE_KEY)
