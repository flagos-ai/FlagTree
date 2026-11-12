# Copyright (c) 2026, The FlagOS Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Load FlagTune's optional Python dependencies with actionable errors."""

from __future__ import annotations

import ctypes
import os
import shutil
import tempfile
import sys
from importlib import import_module
from importlib.util import find_spec, module_from_spec, spec_from_file_location
from pathlib import Path
from threading import RLock
from types import ModuleType


class FlagTuneDependencyError(ImportError):
    """Report an unavailable dependency required by one FlagTune feature."""


def require_optional_dependency(
    module_name: str,
    *,
    distribution_name: str,
    feature: str,
) -> ModuleType:
    """Import one optional package or identify the extra that provides it."""
    try:
        return import_module(module_name)
    except ImportError as exc:
        missing_name = getattr(exc, "name", None) or module_name
        raise FlagTuneDependencyError(f"{feature} requires optional dependency {distribution_name!r} "
                                      f"(failed to import {missing_name!r}); install FlagTree with the "
                                      "'flagtune' extra") from exc


_XGBOOST_LIBRARY = None
_XGBOOST_MODULE = None
_XGBOOST_STORAGE = None
_XGBOOST_LOAD_LOCK = RLock()


def _needs_xgboost_isolation() -> bool:
    if sys.platform != "linux":
        return False
    if "torch_npu" in sys.modules:
        return True
    from triton._flagtree_backend import FLAGTREE_BACKEND
    return FLAGTREE_BACKEND == "ascend"


def _load_private_xgboost(origin: str, feature: str) -> None:
    """Give FlagTune a separate native image when another caller loaded XGBoost."""
    global _XGBOOST_LIBRARY, _XGBOOST_MODULE, _XGBOOST_STORAGE
    storage = tempfile.TemporaryDirectory(prefix="flagtune-xgboost-")
    alias = "_" + Path(storage.name).name.replace("-", "_")
    original = Path(origin).parent
    package = Path(storage.name) / alias
    try:
        shutil.copytree(original, package, ignore=shutil.ignore_patterns("__pycache__"))
        bundled = original.parent / "xgboost.libs"
        if bundled.is_dir():
            (package.parent / "xgboost.libs").symlink_to(bundled, target_is_directory=True)
        library = ctypes.CDLL(str(package / "lib" / "libxgboost.so"), mode=os.RTLD_LOCAL | os.RTLD_DEEPBIND)
        spec = spec_from_file_location(alias, package / "__init__.py", submodule_search_locations=[str(package)])
        module = module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
        # A private module may be finalized before a caller's last model. Keep
        # the C API alive in the destructor instead of looking it up in globals.
        free_booster = sys.modules[alias + ".core"]._LIB.XGBoosterFree

        def release_booster(booster, free=free_booster):
            handle = getattr(booster, "handle", None)
            if handle is not None:
                free(handle)
                booster.handle = None

        module.Booster.__del__ = release_booster
    except Exception as exc:
        for name in list(sys.modules):
            if name == alias or name.startswith(alias + "."):
                del sys.modules[name]
        storage.cleanup()
        raise FlagTuneDependencyError(f"{feature}: cannot create isolated private XGBoost runtime: {exc}") from exc
    _XGBOOST_STORAGE = storage
    _XGBOOST_LIBRARY = library
    _XGBOOST_MODULE = module


def _isolate_xgboost_library(feature: str) -> None:
    """Bind XGBoost before import, avoiding CANN's exported std::regex symbols."""
    global _XGBOOST_LIBRARY
    if _XGBOOST_LIBRARY is not None or not _needs_xgboost_isolation():
        return
    spec = find_spec("xgboost")
    if spec is None:
        return  # Preserve the ordinary missing optional dependency diagnostic.
    if not spec.origin:
        raise FlagTuneDependencyError(f"{feature}: cannot locate the XGBoost package for Ascend symbol isolation")
    library_path = Path(spec.origin).parent / "lib" / "libxgboost.so"
    if not library_path.is_file():
        raise FlagTuneDependencyError(f"{feature}: cannot locate {library_path} for Ascend symbol isolation; "
                                      "install a wheel containing xgboost/lib/libxgboost.so")
    if not hasattr(os, "RTLD_DEEPBIND"):
        raise FlagTuneDependencyError(f"{feature}: Ascend XGBoost symbol isolation requires Linux RTLD_DEEPBIND")
    if "xgboost" in sys.modules:
        _load_private_xgboost(spec.origin, feature)
        return
    try:
        _XGBOOST_LIBRARY = ctypes.CDLL(str(library_path), mode=os.RTLD_LOCAL | os.RTLD_DEEPBIND)
    except OSError as exc:
        raise FlagTuneDependencyError(f"{feature}: cannot load isolated XGBoost library {library_path}: {exc}") from exc


def require_xgboost(feature: str) -> ModuleType:
    """Return XGBoost after validating its scikit-learn ranker dependency."""
    with _XGBOOST_LOAD_LOCK:
        _isolate_xgboost_library(feature)
        xgboost = _XGBOOST_MODULE or require_optional_dependency(
            "xgboost",
            distribution_name="xgboost",
            feature=feature,
        )
        require_optional_dependency(
            "sklearn",
            distribution_name="scikit-learn",
            feature=feature,
        )
        return xgboost
