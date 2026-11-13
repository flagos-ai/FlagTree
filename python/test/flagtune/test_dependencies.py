from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from triton.flagtune import _dependencies
from triton.flagtune._dependencies import FlagTuneDependencyError
from triton.flagtune.runtime import model_loader, model_sources


def _missing_module(name: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(f"No module named {name!r}", name=name)


def test_optional_dependency_error_identifies_package_and_extra(monkeypatch):

    def fail_import(name):
        raise _missing_module(name)

    monkeypatch.setattr(_dependencies, "import_module", fail_import)

    with pytest.raises(
            FlagTuneDependencyError,
            match="requires optional dependency 'numpy'.*'flagtune' extra",
    ) as error:
        _dependencies.require_optional_dependency(
            "numpy",
            distribution_name="numpy",
            feature="FlagTune test feature",
        )

    assert isinstance(error.value.__cause__, ModuleNotFoundError)
    assert error.value.__cause__.name == "numpy"


def test_xgboost_dependency_check_reports_missing_scikit_learn(monkeypatch):
    monkeypatch.setattr(_dependencies, "_needs_xgboost_isolation", lambda: False)

    def import_without_sklearn(name):
        if name == "xgboost":
            return SimpleNamespace()
        raise _missing_module(name)

    monkeypatch.setattr(_dependencies, "import_module", import_without_sklearn)

    with pytest.raises(
            FlagTuneDependencyError,
            match="requires optional dependency 'scikit-learn'",
    ):
        _dependencies.require_xgboost("FlagTune model loading")


def test_download_preserves_dependency_error_from_package_validation(tmp_path, monkeypatch):
    payload = b"downloaded platform package"
    dependency_error = FlagTuneDependencyError("XGBoost is unavailable")

    class Response:

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return payload

    manager = model_loader.FlagTuneModelManager()
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(model_loader, "_open_https", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr(
        model_loader,
        "read_platform_package_bytes",
        lambda *_args, **_kwargs: object(),
    )

    def reject_package(*_args, **_kwargs):
        raise dependency_error

    monkeypatch.setattr(manager, "_validate_package_for_cache", reject_package)
    package = model_sources.RemotePackage(
        "1.0.0",
        "https://models.example.com/nvidia-h20_v1.0.0.tar.gz",
        hashlib.sha256(payload).hexdigest(),
    )

    with pytest.raises(FlagTuneDependencyError) as error:
        manager._download_package("nvidia-h20", package)

    assert error.value is dependency_error
    assert not list(tmp_path.rglob("*.tar.gz"))


@pytest.fixture
def isolated_loader(monkeypatch, tmp_path):
    package = tmp_path / "xgboost"
    library = package / "lib" / "libxgboost.so"
    library.parent.mkdir(parents=True)
    library.touch()
    monkeypatch.setattr(_dependencies, "_XGBOOST_LIBRARY", None)
    monkeypatch.setattr(_dependencies, "_XGBOOST_MODULE", None)
    monkeypatch.setattr(_dependencies, "_needs_xgboost_isolation", lambda: True)
    monkeypatch.delitem(_dependencies.sys.modules, "xgboost", raising=False)
    monkeypatch.setattr(_dependencies, "find_spec", lambda _name: SimpleNamespace(origin=str(package / "__init__.py")))
    monkeypatch.setattr(_dependencies.os, "RTLD_LOCAL", 0, raising=False)
    monkeypatch.setattr(_dependencies.os, "RTLD_DEEPBIND", 8, raising=False)
    return library


def test_xgboost_isolated_before_import_and_handle_reused(monkeypatch, isolated_loader):
    events = []
    handle = object()
    monkeypatch.setattr(_dependencies.ctypes, "CDLL", lambda path, mode: events.append((path, mode)) or handle)
    monkeypatch.setattr(_dependencies, "import_module", lambda name: events.append(name) or SimpleNamespace())
    _dependencies.require_xgboost("test")
    _dependencies.require_xgboost("test")
    assert events == [(str(isolated_loader), 8), "xgboost", "sklearn", "xgboost", "sklearn"]
    assert _dependencies._XGBOOST_LIBRARY is handle


def test_xgboost_isolation_failure_does_not_import_unsafe_library(monkeypatch, isolated_loader):

    def fail_load(*args, **kwargs):
        raise OSError("missing native dependency")

    monkeypatch.setattr(_dependencies.ctypes, "CDLL", fail_load)
    monkeypatch.setattr(_dependencies, "import_module", lambda name: pytest.fail(f"unsafe import: {name}"))
    with pytest.raises(FlagTuneDependencyError, match="cannot load isolated XGBoost") as error:
        _dependencies.require_xgboost("test")
    assert isinstance(error.value.__cause__, OSError)
    assert _dependencies._XGBOOST_LIBRARY is None


def test_xgboost_already_imported_uses_private_runtime(monkeypatch, isolated_loader):
    foreign = SimpleNamespace()
    private = SimpleNamespace()
    monkeypatch.setitem(_dependencies.sys.modules, "xgboost", foreign)
    calls = []

    def load_private(origin, feature):
        calls.append(origin)
        monkeypatch.setattr(_dependencies, "_XGBOOST_LIBRARY", object())
        monkeypatch.setattr(_dependencies, "_XGBOOST_MODULE", private)

    monkeypatch.setattr(_dependencies, "_load_private_xgboost", load_private)
    monkeypatch.setattr(_dependencies, "import_module", lambda name: SimpleNamespace())
    assert _dependencies.require_xgboost("test") is private
    assert _dependencies.require_xgboost("test") is private
    assert _dependencies.sys.modules["xgboost"] is foreign
    assert len(calls) == 1


def test_xgboost_missing_native_library_has_actionable_error(isolated_loader):
    isolated_loader.unlink()
    with pytest.raises(FlagTuneDependencyError, match="cannot locate.*libxgboost.so"):
        _dependencies.require_xgboost("test")


def test_xgboost_missing_deepbind_has_actionable_error(monkeypatch, isolated_loader):
    monkeypatch.delattr(_dependencies.os, "RTLD_DEEPBIND")
    with pytest.raises(FlagTuneDependencyError, match="requires Linux RTLD_DEEPBIND"):
        _dependencies.require_xgboost("test")


def test_xgboost_missing_package_keeps_optional_dependency_error(monkeypatch, isolated_loader):
    monkeypatch.setattr(_dependencies, "find_spec", lambda name: None)

    def missing(name):
        raise _missing_module(name)

    monkeypatch.setattr(_dependencies, "import_module", missing)
    with pytest.raises(FlagTuneDependencyError, match="optional dependency 'xgboost'"):
        _dependencies.require_xgboost("test")


@pytest.mark.parametrize("platform,backend,npu,expected", [
    ("linux", "ascend", False, True),
    ("linux", "", True, True),
    ("linux", "nvidia", False, False),
    ("darwin", "ascend", True, False),
    ("win32", "ascend", True, False),
])
def test_xgboost_isolation_is_scoped_to_linux_ascend(monkeypatch, platform, backend, npu, expected):
    import triton._flagtree_backend as backend_module
    monkeypatch.setattr(_dependencies.sys, "platform", platform)
    monkeypatch.setattr(backend_module, "FLAGTREE_BACKEND", backend)
    if npu:
        monkeypatch.setitem(_dependencies.sys.modules, "torch_npu", SimpleNamespace())
    else:
        monkeypatch.delitem(_dependencies.sys.modules, "torch_npu", raising=False)
    assert _dependencies._needs_xgboost_isolation() is expected


def test_xgboost_concurrent_requests_load_one_library(monkeypatch, isolated_loader):
    from concurrent.futures import ThreadPoolExecutor
    loads = []
    monkeypatch.setattr(_dependencies.ctypes, "CDLL", lambda *a, **kw: loads.append(True) or object())
    monkeypatch.setattr(_dependencies, "import_module", lambda name: SimpleNamespace())
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(_dependencies.require_xgboost, ["test"] * 12))
    assert loads == [True]


def test_private_xgboost_copy_failure_preserves_foreign_module(monkeypatch, isolated_loader):
    foreign = SimpleNamespace()
    monkeypatch.setitem(_dependencies.sys.modules, "xgboost", foreign)

    def fail_copy(*args, **kwargs):
        raise OSError("temporary storage unavailable")

    monkeypatch.setattr(_dependencies.shutil, "copytree", fail_copy)
    with pytest.raises(FlagTuneDependencyError, match="private XGBoost.*temporary storage"):
        _dependencies.require_xgboost("test")
    assert _dependencies.sys.modules["xgboost"] is foreign
    assert _dependencies._XGBOOST_MODULE is None
    assert _dependencies._XGBOOST_LIBRARY is None
