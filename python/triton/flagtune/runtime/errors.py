# Copyright 2026 FlagOS Contributors
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
"""Stable exception categories for FlagTune integration boundaries."""

from contextlib import contextmanager
from functools import wraps


class FlagTuneError(RuntimeError):
    """Base class for failures that FlagGems may handle in AUTO mode."""

    phase = "unknown"
    recoverable_in_auto = True


class PlatformProbeError(FlagTuneError):
    """Device/backend discovery failed before model loading."""

    phase = "preload"


class ModelSourceError(FlagTuneError):
    """Manifest or model package source resolution failed."""

    phase = "preload"


class ModelUnavailableError(FlagTuneError):
    """No model covers the requested platform or identity."""

    phase = "preload"


class ModelValidationError(FlagTuneError):
    """A discovered model package failed validation or loading."""

    phase = "preload"


class ContractExecutionError(FlagTuneError):
    """A loaded model contract cannot serve the current runtime input."""

    phase = "postload"


class ProposerError(FlagTuneError):
    """A loaded proposer failed to produce valid candidates."""

    phase = "postload"


class BenchmarkError(FlagTuneError):
    """A Cost Model candidate failed during benchmark execution."""

    phase = "postload"


@contextmanager
def flagtune_error_boundary(error_type):
    """Normalize ordinary failures without swallowing cancellation or causes."""
    try:
        yield
    except FlagTuneError:
        raise
    except Exception as exc:
        raise error_type(f"{type(exc).__name__}: {exc}") from exc


def flagtune_errors(error_type):
    """Apply a stable error boundary to a synchronous runtime entry point."""

    def decorate(fn):

        @wraps(fn)
        def wrapped(*args, **kwargs):
            with flagtune_error_boundary(error_type):
                return fn(*args, **kwargs)

        return wrapped

    return decorate


__all__ = [
    "flagtune_error_boundary",
    "flagtune_errors",
    "BenchmarkError",
    "ContractExecutionError",
    "FlagTuneError",
    "ModelSourceError",
    "ModelUnavailableError",
    "ModelValidationError",
    "PlatformProbeError",
    "ProposerError",
]
