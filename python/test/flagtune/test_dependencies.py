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
