import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_topk_module():
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "python" / "tutorials" / "tle" / "03-topk.py"
    spec = importlib.util.spec_from_file_location("tle_topk_tutorial", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _set_target(monkeypatch, module, backend: str, arch: str):
    target = SimpleNamespace(backend=backend, arch=arch)
    monkeypatch.setattr(
        module.triton.runtime.driver.active,
        "get_current_target",
        lambda: target,
    )


def test_topk_provider_gfx1201_boundaries(monkeypatch):
    module = _load_topk_module()
    _set_target(monkeypatch, module, "hip", "gfx1201")

    assert module._topk_provider(128, 8) == "triton"
    assert module._topk_provider(8192, 8) == "triton"
    assert module._topk_provider(2048, 32) == "triton"
    assert module._topk_provider(8193, 8) == "radix"
    assert module._topk_provider(8192, 32) == "radix"
    assert module._topk_provider(2049, 32) == "radix"
    assert module._topk_provider(512, 64) == "radix"
    assert module._topk_provider(65536, 2) == "radix"


def test_topk_provider_other_targets_preserve_radix(monkeypatch):
    module = _load_topk_module()

    for backend, arch in (("hip", "gfx942"), ("cuda", "sm90")):
        _set_target(monkeypatch, module, backend, arch)
        assert module._topk_provider(128, 8) == "radix"
