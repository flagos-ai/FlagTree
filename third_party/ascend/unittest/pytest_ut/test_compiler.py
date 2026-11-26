import json
import os
import sys
import warnings
from unittest.mock import MagicMock

import pytest

import triton
import triton.backends.ascend.compiler as compiler
from triton.backends.ascend import utils
from triton.backends.compiler import GPUTarget

pytestmark = pytest.mark.backend("native")


@pytest.fixture(autouse=True)
def reset_option_warnings(monkeypatch):
    monkeypatch.setattr(utils, "_WARNED_DEPRECATED_NPU_OPTIONS", set())


@pytest.fixture(params=["Ascend910B1", "Ascend910_9581"])
def backend(request):
    return compiler.AscendBackend(GPUTarget("npu", request.param, 32))


def _make_torch_npu_mock(cfg_dir):
    """Create a mock torch_npu module whose __file__ is in cfg_dir."""
    mock = MagicMock()
    mock.__file__ = os.path.join(cfg_dir, "__init__.py")
    return mock


def _write_acl_config(cfg_dir, config):
    """Write acl_default.json into cfg_dir."""
    cfg_path = os.path.join(cfg_dir, "acl_default.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f)
    return cfg_path


def test_get_simt_stack_limit_default(monkeypatch):
    """Without torch_npu installed, should return default 1152."""
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    result = compiler.get_simt_stack_limit()
    assert result == 1152


def test_get_simt_stack_limit_from_config(monkeypatch, tmp_path):
    """When acl_default.json has valid simt_stack_size, return it."""
    _write_acl_config(str(tmp_path), {"StackSize": {"simt_stack_size": 2048}})
    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert result == 2048


def test_get_simt_stack_limit_no_StackSize_key(monkeypatch, tmp_path):
    """When acl_default.json has no StackSize key, return default 1152."""
    _write_acl_config(str(tmp_path), {"OtherConfig": {"foo": 1}})
    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert result == 1152


def test_get_simt_stack_limit_no_simt_stack_size_key(monkeypatch, tmp_path):
    """When StackSize exists but simt_stack_size is absent, return default 1152."""
    _write_acl_config(str(tmp_path), {"StackSize": {"other_field": 100}})
    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert result == 1152


def test_get_simt_stack_limit_empty_StackSize(monkeypatch, tmp_path):
    """When StackSize is an empty dict, return default 1152."""
    _write_acl_config(str(tmp_path), {"StackSize": {}})
    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert result == 1152


def test_get_simt_stack_limit_empty_config(monkeypatch, tmp_path):
    """When acl_default.json is an empty dict, return default 1152."""
    _write_acl_config(str(tmp_path), {})
    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert result == 1152


def test_get_simt_stack_limit_file_not_found(monkeypatch, tmp_path):
    """When acl_default.json does not exist, return default 1152."""
    # Do not write acl_default.json; only mock torch_npu.__file__
    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert result == 1152


def test_get_simt_stack_limit_invalid_json(monkeypatch, tmp_path):
    """When acl_default.json is invalid JSON, return default 1152."""
    cfg_path = os.path.join(str(tmp_path), "acl_default.json")
    with open(cfg_path, "w") as f:
        f.write("{invalid json content}")

    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert result == 1152


def test_get_simt_stack_limit_from_config_returns_int(monkeypatch, tmp_path):
    """Return value from config should be an integer."""
    _write_acl_config(str(tmp_path), {"StackSize": {"simt_stack_size": 2048}})
    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert isinstance(result, int)
    assert result == 2048


def test_get_simt_stack_limit_config_overrides_default(monkeypatch, tmp_path):
    """Config value should take precedence over default 1152."""
    _write_acl_config(str(tmp_path), {"StackSize": {"simt_stack_size": 9999}})
    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert result == 9999
    assert result != 1152


def test_empty_options_expose_all_legacy_names(backend):
    options = backend.parse_options({})

    assert utils._DEPRECATED_NPU_OPTIONS <= options.__dict__.keys()
    assert options.__dict__["arch"] == options.arch == backend.target.arch
    assert options.__dict__["warp_size"] == 32
    assert options.__dict__["use_bytecode"] is True
    assert options.__dict__["enable_vf_fusion"] is None


def test_active_options_keep_explicit_values(backend):
    raw = {"enable_mixed_cv": True, "enable_dynamic_cv_pipeline": False, "num_warps": 16}

    options = backend.parse_options(raw)

    assert options.enable_mixed_cv is True
    assert options.enable_dynamic_cv_pipeline is False
    assert options.num_warps == 16


_IGNORED_OPTIONS = sorted(utils._DEPRECATED_NPU_OPTIONS - utils._DEPRECATED_NPU_OPTION_ROUTES.keys() -
                          utils._DEPRECATED_NPU_OPTION_ALIASES.keys() - {"stream"})


@pytest.mark.parametrize("name", _IGNORED_OPTIONS)
def test_ignored_values_preserve_defaults_and_input(backend, name):
    baseline = backend.parse_options({})
    raw = {name: object()}
    original = dict(raw)

    with pytest.warns(FutureWarning, match=name):
        options = backend.parse_options(raw)

    assert raw == original
    assert options.__dict__ == baseline.__dict__
    assert options.hash() == baseline.hash()


@pytest.mark.parametrize("value", [None, "", "--unsupported-bisheng-option"])
def test_legacy_bisheng_options_launch_uses_backend_defaults(backend, value):

    @triton.jit
    def kernel():
        pass

    baseline = backend.parse_options({})
    kwargs = {"bisheng_options": value}

    with pytest.warns(FutureWarning, match="bisheng_options.*backend-managed"):
        options, signature, constexprs, attrs = kernel._pack_args(backend, kwargs, {}, [], None)

    assert kwargs == {"bisheng_options": value}
    assert options.bisheng_options == "-cce-link-aicore-ll-module " + compiler.get_libdevice()
    assert options.__dict__ == baseline.__dict__
    assert options.hash() == baseline.hash()
    assert signature == constexprs == attrs == {}


@pytest.mark.parametrize("stream", [None, 0, object()], ids=["none", "zero", "unused-object"])
def test_legacy_stream_is_consumed_without_losing_option_discovery(backend, stream):
    baseline = backend.parse_options({"num_warps": 16})
    raw = {"stream": stream, "num_warps": 16, "kernel_arg": 42}

    with pytest.warns(FutureWarning, match="stream"):
        options = backend.parse_options(raw)

    assert raw == {"num_warps": 16, "kernel_arg": 42}
    assert options.__dict__["stream"] is None
    assert options.__dict__ == baseline.__dict__
    assert options.hash() == baseline.hash()


@pytest.mark.parametrize(
    "raw, expected",
    [
        ({"force_simt_only": True, "compile_mode": "simd"}, {"compile_mode": "simt_only"}),
        ({"force_simt_only": True, "force_simt_template": True}, {"compile_mode": "simt_only"}),
        ({"force_simt_template": True}, {"compile_mode": "simd_simt_template"}),
        ({"force_simt_only": False, "compile_mode": "simd"}, {"compile_mode": "simd"}),
        ({"intra_cache_num": 4}, {"buf_slot_num_of_veccore": 4}),
        ({"inter_cache_num": 3}, {"buf_slot_num_of_crosscore": 3}),
        ({"load_cache_num": 2}, {"buf_slot_num_of_gm": 2}),
        ({"intra_cache_num": 4, "buf_slot_num_of_veccore": 6}, {"buf_slot_num_of_veccore": 6}),
        ({"enable_bishengir_simt_optimization": 101, "compile_mode": "simt_only"
          }, {"simt_optimization_mode": 101, "compile_mode": "simt_only"}),
        ({"enable_bishengir_simt_optimization": 101, "simt_optimization_mode": 100, "compile_mode": "simt_only"
          }, {"simt_optimization_mode": 100, "compile_mode": "simt_only"}),
    ],
)
def test_legacy_routes_preserve_precedence_and_input(raw, expected):
    backend = compiler.AscendBackend(GPUTarget("npu", "Ascend910_9581", 32))
    original = dict(raw)
    baseline = backend.parse_options(expected)

    with pytest.warns(FutureWarning):
        options = backend.parse_options(raw)

    assert raw == original
    assert options.__dict__ == baseline.__dict__
    assert options.hash() == baseline.hash()


def test_simt_alias_obeys_target_and_mode_restrictions(backend):
    raw = {"enable_bishengir_simt_optimization": 101}

    with pytest.warns((FutureWarning, UserWarning)) as caught:
        options = backend.parse_options(raw)

    assert any("only takes effect" in str(warning.message) for warning in caught)
    assert options.simt_optimization_mode == 0
    assert raw == {"enable_bishengir_simt_optimization": 101}


def test_internal_grid_value_survives_serialization_and_reparse(backend):
    raw = {"grid_num_tiles": utils._InternalNPUOptionInt(8)}
    options = backend.parse_options(raw)
    serialized = json.loads(json.dumps(options.__dict__))
    # JITFunction.preload restores tuple-valued options after JSON decoding.
    restored = {name: tuple(value) if isinstance(value, list) else value for name, value in serialized.items()}
    original = dict(restored)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reparsed = backend.parse_options(restored)

    assert not caught
    assert restored == original
    assert reparsed.grid_num_tiles == options.grid_num_tiles == 8
    assert reparsed.__dict__ == options.__dict__
    assert reparsed.hash() == options.hash()


def test_unknown_options_remain_available_for_jit_validation(backend):
    raw = {"unknown_compile_option": True}

    options = backend.parse_options(raw)

    assert raw == {"unknown_compile_option": True}
    assert "unknown_compile_option" not in options.__dict__
