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
import sys
from importlib import import_module
from importlib.util import find_spec
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
_XGBOOST_LOAD_LOCK = RLock()


def _needs_xgboost_isolation() -> bool:
    if sys.platform != "linux":
        return False
    if "torch_npu" in sys.modules:
        return True
    from triton._flagtree_backend import FLAGTREE_BACKEND
    return FLAGTREE_BACKEND == "ascend"


def _isolate_xgboost_library(feature: str) -> None:
    """Bind XGBoost before import, avoiding CANN's exported std::regex symbols."""
    global _XGBOOST_LIBRARY
    if _XGBOOST_LIBRARY is not None or not _needs_xgboost_isolation():
        return
    if "xgboost" in sys.modules:
        raise FlagTuneDependencyError(
            f"{feature}: XGBoost was already imported without FlagTune's Ascend symbol isolation; "
            "restart the process and call require_xgboost() before importing xgboost")
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
    try:
        _XGBOOST_LIBRARY = ctypes.CDLL(str(library_path), mode=os.RTLD_LOCAL | os.RTLD_DEEPBIND)
    except OSError as exc:
        raise FlagTuneDependencyError(f"{feature}: cannot load isolated XGBoost library {library_path}: {exc}") from exc


def require_xgboost(feature: str) -> ModuleType:
    """Return XGBoost after validating its scikit-learn ranker dependency."""
    with _XGBOOST_LOAD_LOCK:
        _isolate_xgboost_library(feature)
        xgboost = require_optional_dependency(
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
