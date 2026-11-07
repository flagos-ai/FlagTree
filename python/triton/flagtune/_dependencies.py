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

from importlib import import_module
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


def require_xgboost(feature: str) -> ModuleType:
    """Return XGBoost after validating its scikit-learn ranker dependency."""
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
