"""Test CommonIR capability discovery and the runtime gate."""
import importlib.util
from types import SimpleNamespace

import pytest


def _load_capability(monkeypatch, binding):
    from triton import _C, _common_ir

    # Mock capability discovery while retaining the native environment reader.
    monkeypatch.setattr(_C, "libtriton", binding)
    path = _common_ir.__file__
    spec = importlib.util.spec_from_file_location("_test_common_ir_capability", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("binding", [
    SimpleNamespace(),
    SimpleNamespace(tle=SimpleNamespace()),
    SimpleNamespace(iluvatar=SimpleNamespace()),
])
def test_missing_common_ir_capability_is_disabled(monkeypatch, binding):
    monkeypatch.setenv("FLAGTREE_COMMON_IR", "1")
    module = _load_capability(monkeypatch, binding)
    assert module.AVAILABLE is False
    assert module.ENABLED is False


@pytest.mark.parametrize("plugin", ["tle", "iluvatar"])
@pytest.mark.parametrize("enabled", [False, True])
def test_common_ir_capability_is_queried_once(monkeypatch, plugin, enabled):
    monkeypatch.setenv("FLAGTREE_COMMON_IR", "1")
    calls = []

    def query():
        calls.append(True)
        return enabled

    module = _load_capability(monkeypatch, SimpleNamespace(**{plugin: SimpleNamespace(is_common_ir_enabled=query)}))
    assert module.ENABLED is enabled
    assert module.ENABLED is enabled
    assert len(calls) == 1


@pytest.mark.parametrize("available", [False, True])
def test_tle_capability_is_preferred_over_vendor(monkeypatch, available):
    monkeypatch.setenv("FLAGTREE_COMMON_IR", "1")

    def tle_query():
        return available

    def vendor_query():
        raise AssertionError("vendor query should not run when tle is present")

    module = _load_capability(
        monkeypatch,
        SimpleNamespace(tle=SimpleNamespace(is_common_ir_enabled=tle_query),
                        iluvatar=SimpleNamespace(is_common_ir_enabled=vendor_query)))
    assert module.ENABLED is available


@pytest.mark.parametrize("plugin", ["tle", "iluvatar"])
@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("setting, runtime_enabled", [(None, True), ("0", False), ("1", True)])
def test_common_ir_runtime_gate(monkeypatch, plugin, available, setting, runtime_enabled):
    if setting is None:
        monkeypatch.delenv("FLAGTREE_COMMON_IR", raising=False)
    else:
        monkeypatch.setenv("FLAGTREE_COMMON_IR", setting)
    binding = SimpleNamespace(**{plugin: SimpleNamespace(is_common_ir_enabled=lambda: available)})
    module = _load_capability(monkeypatch, binding)
    assert module.AVAILABLE is available
    assert module.ENABLED is (available and runtime_enabled)


@pytest.mark.parametrize("plugin", ["tle", "iluvatar"])
def test_common_ir_query_errors_are_not_hidden(monkeypatch, plugin):

    def query():
        raise RuntimeError("broken CommonIR capability query")

    with pytest.raises(RuntimeError, match="broken CommonIR capability query"):
        _load_capability(monkeypatch, SimpleNamespace(**{plugin: SimpleNamespace(is_common_ir_enabled=query)}))
