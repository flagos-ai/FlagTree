# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Exercise structured modes through Config, JSON caches and autotune without a device."""
import ast
import itertools
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

pytestmark = pytest.mark.backend("none")
ROOT = Path(__file__).resolve().parents[4]
BACKEND = ROOT / "third_party" / "ascend" / "backend"
MODE = {"gm": 4, "l1": 2, "l0c": 1, "ub": 2}
MODE_PAIRS = tuple(sorted(MODE.items()))


def _load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [
        node for node in tree.body if getattr(node, "name", None) in names or (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names for target in node.targets))
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


@pytest.fixture
def tuning(monkeypatch):
    namespace = dict(Any=Any, Dict=Dict, List=List, Optional=Optional, itertools=itertools,
                     _remove_deprecated_npu_options=lambda value: dict(value))
    _load_definitions(ROOT / "python/triton/runtime/autotuner.py", {"Config"}, namespace)
    utils = ModuleType("triton.backends.ascend.utils")
    _load_definitions(BACKEND / "utils.py", {"_multibuffer_mode_to_tuple"}, utils.__dict__)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    namespace["_multibuffer_mode_to_tuple"] = utils._multibuffer_mode_to_tuple
    runtime = ModuleType("triton.runtime.autotuner")
    runtime.Config = namespace["Config"]
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    _load_definitions(BACKEND / "runtime/__init__.py", {"_patch_config_init"}, namespace)
    namespace["_patch_config_init"]()
    names = {
        "_ALL_PARAMS",
        "_DEFAULTS",
        "_VALID_VALUES",
        "_CUBE_PARAMS",
        "_MIXCV_PARAMS",
        "_VECTOR_PARAMS",
        "_VALIDATION_RULES",
        "_check_boolean_list",
        "_check_string_in_set",
        "_check_int_in_set",
        "_check_multibuffer_mode_list",
        "BaseAutotuner",
        "CubeAutotuner",
        "MixcvAutotuner",
        "VectorAutotuner",
        "get_max_configs",
        "_normalize_config_hints",
        "_multibuffer_to_num_stages",
        "_DEFAULT_HINT_NUM_STAGES",
        "_clone_config_with_kwargs",
        "_expand_configs_with_hints",
        "_validate_config_hint_values",
        "_DEFAULT_COMPILE_MODE",
    }
    _load_definitions(BACKEND / "runtime/autotuner.py", names, namespace)
    namespace["_get_constexpr_candidates_from_fn"] = lambda _fn: []
    namespace["get_byte_per_numel"] = lambda _dtype: 4
    tree = ast.parse((BACKEND / "runtime/autotuner.py").read_text())
    tuner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AutoTilingTuner")
    generate = next(node for node in tuner.body if getattr(node, "name", None) == "generate_key_and_configs")
    exec(compile(ast.Module(body=[generate], type_ignores=[]), str(BACKEND / "runtime/autotuner.py"), "exec"),
         namespace)
    return SimpleNamespace(**namespace)


def test_multibuffer_dictionary_config_survives_json_cache(tuning):
    config = tuning.Config({"multibuffer_mode": MODE})
    assert config.num_stages is None
    assert "num_stages" not in config.all_kwargs()
    assert config.kwargs["multibuffer_mode"] == MODE_PAIRS
    restored = tuning.Config(**json.loads(json.dumps(config.__dict__)))
    assert restored == config
    assert hash(restored) == hash(config)


@pytest.mark.parametrize("name", ["CubeAutotuner", "MixcvAutotuner", "VectorAutotuner"])
def test_multibuffer_mode_presets_keep_existing_options(tuning, name):
    generator = getattr(tuning, name)
    modes = [MODE, {"future": 0}]
    configs = generator.get_configs(multibuffer_mode=modes)
    assert configs and {config.kwargs["multibuffer_mode"]
                        for config in configs} == {tuple(sorted(mode.items()))
                                                   for mode in modes}
    assert all(config.num_stages is None for config in configs)
    baseline = {tuple(sorted(config.kwargs.items())) for config in generator.get_configs()}
    assert {
        tuple(sorted((key, value)
                     for key, value in config.kwargs.items()
                     if key != "multibuffer_mode"))
        for config in configs
    } == baseline


def test_multibuffer_legacy_presets_keep_their_defaults(tuning):
    configs = tuning.MixcvAutotuner.get_configs()
    assert len(configs) == 8
    assert all(config.num_stages == 2 and "multibuffer_mode" not in config.kwargs for config in configs)
    assert {config.kwargs["set_workspace_multibuffer"] for config in configs} == {2, 4}
    assert all(config.kwargs["limit_auto_multi_buffer_of_local_buffer"] == "no-l0c" for config in configs)
    assert all(config.kwargs["limit_auto_multi_buffer_only_for_local_buffer"] is False for config in configs)


def test_multibuffer_explicit_legacy_preset_values_survive(tuning):
    configs = tuning.MixcvAutotuner.get_configs(multibuffer_mode=[MODE], set_workspace_multibuffer=[2, 4],
                                                limit_auto_multi_buffer_of_local_buffer=["no-l0c"])
    assert configs and {config.kwargs["set_workspace_multibuffer"] for config in configs} == {2, 4}
    assert all(config.kwargs["limit_auto_multi_buffer_of_local_buffer"] == "no-l0c" for config in configs)


@pytest.mark.parametrize("mode_in_base", [False, True])
def test_multibuffer_max_configs_keep_existing_options(tuning, mode_in_base):
    base = tuning.Config({"multibuffer_mode": MODE} if mode_in_base else {}, num_stages=None)
    kwargs = {} if mode_in_base else {"multibuffer_mode": [MODE]}
    configs = tuning.get_max_configs(base, kernel_type="mixcv", **kwargs)
    assert configs and all(config.kwargs["multibuffer_mode"] == MODE_PAIRS for config in configs)
    assert all(config.num_stages is None for config in configs)
    baseline = tuning.get_max_configs(tuning.Config({}, num_stages=2), kernel_type="mixcv")
    assert {
        tuple(sorted((key, value)
                     for key, value in config.kwargs.items()
                     if key != "multibuffer_mode"))
        for config in configs
    } == {tuple(sorted(config.kwargs.items()))
          for config in baseline}


def test_multibuffer_max_configs_preserve_explicit_old_values(tuning):
    base = tuning.Config({"multibuffer_mode": MODE, "limit_auto_multi_buffer_of_local_buffer": "no-l0c"})
    configs = tuning.get_max_configs(base, kernel_type="mixcv", set_workspace_multibuffer=[2, 4])
    assert {config.kwargs["set_workspace_multibuffer"] for config in configs} == {2, 4}
    assert all(config.kwargs["limit_auto_multi_buffer_of_local_buffer"] == "no-l0c" for config in configs)


@pytest.mark.parametrize("mode_in_base", [False, True])
def test_multibuffer_hints_preserve_explicit_switch(tuning, mode_in_base):
    hints = {"multibuffer": [False, True]}
    if not mode_in_base:
        hints["multibuffer_mode"] = [MODE]
    assert tuning._normalize_config_hints(hints, inject_default_num_stages=True,
                                          has_multibuffer_mode=mode_in_base) == hints


def test_multibuffer_legacy_hints_keep_stage_expansion(tuning):
    assert tuning._normalize_config_hints({"multibuffer": [False, True]}) == {"num_stages": [1, 2]}


@pytest.mark.parametrize("value,valid", [
    ([MODE], True),
    ([list(MODE_PAIRS)], True),
    ([{"future": -1}], True),
    ([{}], True),
    (["[(gm,2)]"], False),
    ([{"gm": 2.0}], False),
    ([{"gm": True}], False),
    ([{1: 2}], False),
    ([None], False),
    ([], False),
])
def test_multibuffer_tuning_values_require_string_keys_and_integer_counts(tuning, value, valid):
    assert tuning._VALIDATION_RULES["multibuffer_mode"]["check"](value, "multibuffer_mode") is valid


def test_multibuffer_dictionary_is_copied_for_config_hashing(tuning):
    mode = dict(MODE)
    config = tuning.Config({"multibuffer_mode": mode})
    before = hash(config)
    mode["gm"] = 99
    assert config.kwargs["multibuffer_mode"] == MODE_PAIRS
    assert hash(config) == before


@pytest.mark.parametrize("stages", [0, 1, 2, 3])
@pytest.mark.parametrize("mode", [MODE, {}])
def test_multibuffer_config_rejects_num_stages(tuning, stages, mode):
    with pytest.raises(ValueError, match="num_stages and multibuffer_mode cannot be specified together"):
        tuning.Config({"multibuffer_mode": mode}, num_stages=stages)


def test_multibuffer_config_default_and_explicit_none(tuning):
    assert tuning.Config({}).num_stages == 3
    assert tuning.Config({}, num_stages=None).num_stages is None
    assert tuning.Config({"multibuffer_mode": MODE}, num_stages=None).num_stages is None


@pytest.mark.parametrize("name", ["CubeAutotuner", "MixcvAutotuner", "VectorAutotuner"])
def test_multibuffer_presets_reject_explicit_num_stages(tuning, name):
    with pytest.raises(ValueError, match="num_stages and multibuffer_mode cannot be specified together"):
        getattr(tuning, name).get_configs(multibuffer_mode=[MODE], num_stages=[2])


@pytest.mark.parametrize("mode_in_base", [False, True])
def test_multibuffer_max_configs_reject_explicit_num_stages(tuning, mode_in_base):
    if mode_in_base:
        base = tuning.Config({"multibuffer_mode": MODE})
        kwargs = {"num_stages": [2]}
    else:
        base = tuning.Config({}, num_stages=2)
        kwargs = {"multibuffer_mode": [MODE]}
    with pytest.raises(ValueError, match="num_stages and multibuffer_mode cannot be specified together"):
        tuning.get_max_configs(base, **kwargs)


@pytest.mark.parametrize("mode_in_base", [False, True])
def test_multibuffer_hints_reject_explicit_num_stages(tuning, mode_in_base):
    hints = {"num_stages": [2]}
    if not mode_in_base:
        hints["multibuffer_mode"] = [MODE]
    with pytest.raises(ValueError, match="num_stages and multibuffer_mode cannot be specified together"):
        tuning._normalize_config_hints(hints, has_multibuffer_mode=mode_in_base)


def _auto_tuner(tuning, hints=None, user_configs=(), generated_configs=()):
    tuner = SimpleNamespace(
        arg_names=["x"],
        keys=[],
        fn=object(),
        cache={},
        axis_arg_names={},
        _raw_config_hints=hints or {},
        user_configs=list(user_configs),
        gen_configs=[],
        auto_gen_config=True,
        _autoparse_axis_params=lambda _args: None,
    )
    tuner._gen_tile_configs = lambda *_args: setattr(tuner, "gen_configs", list(generated_configs))
    return tuner


@pytest.mark.parametrize("mode_in_hints", [False, True])
@pytest.mark.parametrize("has_generated_config", [False, True])
def test_multibuffer_auto_configs_drop_generated_stage_defaults(tuning, mode_in_hints, has_generated_config):
    hints = {"multibuffer_mode": [MODE]} if mode_in_hints else {}
    launch = {} if mode_in_hints else {"multibuffer_mode": MODE}
    generated = [tuning.Config({}, num_stages=1)] if has_generated_config else []
    tuner = _auto_tuner(tuning, hints=hints, generated_configs=generated)
    tuning.generate_key_and_configs(tuner, SimpleNamespace(dtype="float32"), **launch)
    assert tuner.configs
    for config in tuner.configs:
        options = dict(config.all_kwargs(), **launch)
        assert dict(options["multibuffer_mode"]) == MODE
        assert "num_stages" not in options


@pytest.mark.parametrize("source", ["launch", "hints", "config_mode", "config_stages"])
def test_multibuffer_auto_configs_reject_explicit_conflicts_before_generation(tuning, source):
    launch, hints, configs = {}, {}, []
    if source == "launch":
        launch = {"num_stages": 2, "multibuffer_mode": MODE}
    elif source == "hints":
        hints = {"num_stages": [2], "multibuffer_mode": [MODE]}
    elif source == "config_mode":
        configs = [tuning.Config({"multibuffer_mode": MODE})]
        launch = {"num_stages": 2}
    else:
        configs = [tuning.Config({}, num_stages=2)]
        launch = {"multibuffer_mode": MODE}
    tuner = _auto_tuner(tuning, hints=hints, user_configs=configs)

    def unexpected_generation(*_args):
        raise AssertionError("Conflicting options reached tile generation")

    tuner._gen_tile_configs = unexpected_generation
    with pytest.raises(ValueError, match="num_stages and multibuffer_mode cannot be specified together"):
        tuning.generate_key_and_configs(tuner, SimpleNamespace(dtype="float32"), **launch)


def test_multibuffer_launch_mode_prevents_default_stage_hints(tuning):
    tuner = _auto_tuner(tuning, hints={"enable_ubuf_saving": [True]}, user_configs=[tuning.Config({}, num_stages=None)])
    tuning.generate_key_and_configs(tuner, SimpleNamespace(dtype="float32"), multibuffer_mode=MODE)
    assert "num_stages" not in tuner.config_hints
    assert all(config.num_stages is None for config in tuner.configs)


def test_multibuffer_dictionary_order_does_not_change_config_hash(tuning):
    config = tuning.Config({"multibuffer_mode": MODE})
    reordered = tuning.Config({"multibuffer_mode": dict(reversed(list(MODE.items())))})
    assert reordered == config
    assert hash(reordered) == hash(config)
    assert {config: "cached"}[reordered] == "cached"


def test_multibuffer_dictionary_order_does_not_change_autotune_key(tuning):
    tuner = _auto_tuner(tuning)
    arg = SimpleNamespace(dtype="float32")
    key = tuning.generate_key_and_configs(tuner, arg, multibuffer_mode=MODE)
    reordered = tuning.generate_key_and_configs(tuner, arg, multibuffer_mode=dict(reversed(list(MODE.items()))))
    assert {key: "cached"}[reordered] == "cached"
    changed = tuning.generate_key_and_configs(tuner, arg, multibuffer_mode={**MODE, "ub": 8})
    assert changed != key
