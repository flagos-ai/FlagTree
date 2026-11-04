# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Define the installed FlagTune package version.

``__version__`` is sent in model package download user agents and is compared
with optional ``flagtune_version_min`` and ``flagtune_version_max`` fields in
an archive's YAML contract by :class:`triton.flagtune.runtime.model_loader.FlagTuneModelManager`.
It is the runtime implementation compatibility version, not the independently
versioned model artifact revision stored in each archive path.
"""

import logging
from collections.abc import Mapping
from typing import Any

__version__ = "1.1.0"
EXPLICIT_CONFIGS_VERSION = "1.1.0"


def _compat_version(version: str) -> tuple:
    """Retain the numeric compatibility-bound comparison used by 1.0."""
    parts = []
    for part in str(version).split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    return tuple(parts + [0] * max(0, 3 - len(parts)))


def validate_model_runtime(config: Mapping[str, Any], *, runtime_version=None, source="model config") -> None:
    """Check requirements before interpreting version-specific model fields.

    Published explicit-config bundles without a minimum-version declaration
    still require the runtime that introduced that feature. Artifact bytes and
    their bound digests are never rewritten to add compatibility metadata.
    """
    current = __version__ if runtime_version is None else runtime_version
    minimums = [config.get("flagtune_version_min")]
    if "configs" in config:
        minimums.append(EXPLICIT_CONFIGS_VERSION)
    for minimum in minimums:
        if minimum and _compat_version(minimum) > _compat_version(current):
            raise ValueError(f"Model at {source} requires FlagTune >= {minimum}, current version is {current}")
    maximum = config.get("flagtune_version_max")
    if maximum and _compat_version(maximum) < _compat_version(current):
        logging.getLogger(__name__).warning("Model at %s was built for FlagTune <= %s, current version is %s", source,
                                            maximum, current)
