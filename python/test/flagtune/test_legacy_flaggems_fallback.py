"""Exercise the legacy call protocol without importing or patching FlagGems."""

from types import SimpleNamespace

import pytest
import triton

from triton.flagtune.runtime import proposer
from triton.flagtune.runtime.errors import ModelUnavailableError, ModelValidationError


@pytest.fixture
def legacy(monkeypatch):
    monkeypatch.delenv("USE_FLAGTUNE_COST_MODEL", raising=False)
    monkeypatch.delenv("USE_FLAGTUNE", raising=False)
    calls = []

    def missing(*args, **kwargs):
        calls.append((args, kwargs))
        raise ModelUnavailableError("test model missing")

    monkeypatch.setattr(proposer, "_MODEL_MANAGER", SimpleNamespace(load=missing))
    namespace = {
        "__name__": "flag_gems.utils.libentry",
        "api": proposer,
        "identity": dict(op_id="flaggems/mm", variant="gemv", platform_key="nvidia-h20", dtype_key="f32-f32-f32"),
    }
    # Compile the exact old call shape so detection exercises real Python
    # frames. No installed FlagGems version or filesystem path is required.
    exec(
        "def _ensure_flagtune_proposer():\n"
        "    loaded = api.load_model_bundle(**identity)\n"
        "    return loaded, api.make_config_proposer(**identity)\n"
        "def flagtune_policy(self):\n"
        "    return _ensure_flagtune_proposer()\n",
        namespace,
    )
    hook = lambda args: None
    configs = [triton.Config({"BLOCK_SIZE": 32}, pre_hook=hook), triton.Config({"BLOCK_SIZE": 64})]
    tuner = SimpleNamespace(_flagtune_op_id="flaggems/mm", _flagtune_variant="gemv", configs=configs)
    return namespace, tuner, calls


def _project(config):
    return dict(config.kwargs, num_warps=config.num_warps, num_stages=config.num_stages, num_ctas=config.num_ctas)


def test_legacy_missing_model_selects_one_original_config(legacy):
    namespace, tuner, calls = legacy
    with pytest.warns(RuntimeWarning, match="without Cost Model prediction"):
        loaded, propose = namespace["flagtune_policy"](tuner)
    assert not calls, "legacy compatibility must bypass model loading"
    assert loaded.model_version == "legacy-single-config"
    assert loaded.variant.param_names == ["BLOCK_SIZE"]
    assert loaded.variant.normalize_inputs({}) == {}

    def unexpected_benchmark(*args):
        pytest.fail("the compatibility proposer must not benchmark")

    initial = [_project(c) for c in tuner.configs]
    assert propose(unexpected_benchmark, {}, initial, {}) == initial[:1]
    selected = loaded.variant.to_config(initial[0])
    assert selected is tuner.configs[0]
    assert selected.pre_hook is tuner.configs[0].pre_hook

    # A cached adapter must use the next call's first candidate, not capture
    # the first-ever shape/config or discard its hook during conversion.
    tuner.configs = [tuner.configs[1], tuner.configs[0]]
    initial = [_project(c) for c in tuner.configs]
    assert loaded.variant.to_config(propose(None, {}, initial, {})[0]) is tuner.configs[0]
    with pytest.raises(ValueError, match="empty"):
        propose(None, {}, [], {})


@pytest.mark.parametrize("setting", ["0", "invalid"])
def test_legacy_ordinary_call_is_compatible(legacy, monkeypatch, setting):
    namespace, tuner, _ = legacy
    monkeypatch.setenv("USE_FLAGTUNE_COST_MODEL", setting)
    with pytest.warns(RuntimeWarning, match="without Cost Model prediction"):
        loaded, _ = namespace["flagtune_policy"](tuner)
    assert loaded.model_version == "legacy-single-config"


def test_legacy_explicit_cost_model_keeps_original_error(legacy, monkeypatch):
    namespace, tuner, _ = legacy
    monkeypatch.setenv("USE_FLAGTUNE_COST_MODEL", "1")
    with pytest.raises(ModelUnavailableError, match="test model missing"):
        namespace["flagtune_policy"](tuner)


def test_direct_and_new_callers_keep_original_error(legacy):
    namespace, tuner, _ = legacy
    with pytest.raises(ModelUnavailableError):
        proposer.load_model_bundle(**namespace["identity"])
    namespace["__name__"] = "flag_gems.flagtune.cost_model"
    with pytest.raises(ModelUnavailableError):
        namespace["flagtune_policy"](tuner)


def test_legacy_validation_failure_is_not_hidden(legacy, monkeypatch):
    namespace, tuner, _ = legacy
    monkeypatch.setenv("USE_FLAGTUNE_COST_MODEL", "1")

    def invalid(*args, **kwargs):
        raise ModelValidationError("invalid model archive")

    monkeypatch.setattr(proposer, "_MODEL_MANAGER", SimpleNamespace(load=invalid))
    with pytest.raises(ModelValidationError, match="invalid model archive"):
        proposer._get_model_manager().load("flaggems/mm", "gemv", platform_key="nvidia-h20", dtype_key="f32-f32-f32")


def test_legacy_existing_model_is_unchanged(legacy, monkeypatch):
    namespace, tuner, _ = legacy
    sentinel = object()
    monkeypatch.setattr(proposer, "_MODEL_MANAGER", SimpleNamespace(load=lambda *a, **k: sentinel))
    exec("def flagtune_policy(self):\n    return api.load_model_bundle(**identity)\n", namespace)
    assert namespace["flagtune_policy"](tuner) is sentinel
