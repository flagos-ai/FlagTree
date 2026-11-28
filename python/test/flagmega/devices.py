# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Device predicates shared by FlagMega GPU tests."""

import pytest

NVIDIA_SM90_REASON = "requires an NVIDIA SM90 GPU"


def is_nvidia_sm90() -> bool:
    try:
        import torch
    except ImportError:
        return False
    # ROCm torch also reports device type "cuda" and returns the gfx version as
    # the capability, e.g. (9, 0) on gfx90a, so the build must be checked too.
    return (torch.cuda.is_available() and not torch.version.hip and torch.cuda.get_device_capability() == (9, 0))


def require_nvidia_sm90():
    if not is_nvidia_sm90():
        pytest.skip(NVIDIA_SM90_REASON)


requires_nvidia_sm90 = pytest.mark.skipif(not is_nvidia_sm90(), reason=NVIDIA_SM90_REASON)
