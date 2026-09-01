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
"""Self-contained model integration for FlagTune configuration prediction.

Runtime callers identify a model bundle by
``(platform_key, op_id, variant, dtype_key)``. The bundle supplies
the parameter space, input rules, safe feature expressions, version metadata,
and XGBoost model without prior operator registration::

    from triton.flagtune import make_config_proposer

    proposer = make_config_proposer(
        "flaggems/mm",
        "general_tma",
        platform_key="nvidia-h800-80gb-hbm3",
        dtype_key="bf16-bf16-bf16",
    )
"""

import os

from triton.flagtune._version import __version__
from triton.flagtune.core.interfaces import BenchmarkFn, ConfigProposer
from triton.flagtune.runtime.autotuner import Flagtuner, flagtune

_ENABLED = None


def is_enabled() -> bool:
    """Return whether Triton's FlagTune integration is enabled for this process.

    ``FLAGTUNE_ENABLE`` must be exactly ``"1"`` after whitespace stripping.
    The result is cached on first access. This switch remains independent from
    FlagGems' independent ``FLAGGEMS_FLAGTUNE_EXPANDED`` expanded-config
    control.
    """
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("FLAGTUNE_ENABLE", "").strip() == "1"
    return _ENABLED


def load_model_bundle(
    op_id: str,
    variant: str,
    *,
    platform_key: str,
    dtype_key: str,
    model_version=None,
):
    """Load the self-contained runtime bundle for an exact operator variant.

    Arguments and exceptions are forwarded lazily to
    :func:`triton.flagtune.runtime.proposer.load_model_bundle`, avoiding XGBoost imports
    until a model is actually requested.
    """
    from triton.flagtune.runtime.proposer import load_model_bundle as _load

    return _load(
        op_id,
        variant,
        platform_key=platform_key,
        dtype_key=dtype_key,
        model_version=model_version,
    )


def make_config_proposer(
    op_id: str,
    variant: str,
    *,
    platform_key: str,
    dtype_key: str,
    model_version=None,
) -> ConfigProposer:
    """Create an XGBoost/GA proposer for an exact operator variant.

    Resolution, YAML compilation, config/model digest validation, and XGBoost
    loading happen during this call. Their errors propagate to the caller so
    integration layers can apply their normal fallback policy.
    """
    from triton.flagtune.runtime.proposer import make_config_proposer as _make

    return _make(
        op_id,
        variant,
        platform_key=platform_key,
        dtype_key=dtype_key,
        model_version=model_version,
    )
