"""CommonIR capability discovery must also work with vendor-only bindings."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


def _load_capability(monkeypatch, binding):
    extension = ModuleType("triton._C")
    extension.libtriton = binding
    monkeypatch.setitem(sys.modules, "triton._C", extension)
    path = Path(__file__).resolve().parents[2] / "triton" / "_common_ir.py"
    spec = importlib.util.spec_from_file_location("_test_common_ir_capability", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("binding", [SimpleNamespace(), SimpleNamespace(tle=SimpleNamespace())])
def test_missing_common_ir_capability_is_disabled(monkeypatch, binding):
    assert _load_capability(monkeypatch, binding).ENABLED is False


@pytest.mark.parametrize("enabled", [False, True])
def test_common_ir_capability_is_queried_once(monkeypatch, enabled):
    calls = []

    def query():
        calls.append(True)
        return enabled

    module = _load_capability(monkeypatch, SimpleNamespace(tle=SimpleNamespace(is_common_ir_enabled=query)))
    assert module.ENABLED is enabled
    assert module.ENABLED is enabled
    assert len(calls) == 1


def test_common_ir_query_errors_are_not_hidden(monkeypatch):

    def query():
        raise RuntimeError("broken CommonIR capability query")

    with pytest.raises(RuntimeError, match="broken CommonIR capability query"):
        _load_capability(monkeypatch, SimpleNamespace(tle=SimpleNamespace(is_common_ir_enabled=query)))
