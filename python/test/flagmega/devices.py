# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Device predicates shared by FlagMega GPU tests."""

import pytest

NVIDIA_SM90_REASON = "requires an NVIDIA SM90 GPU"


def _cuda_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


def nvidia_capability(device=0):
    """The NVIDIA compute capability of ``device``, or None off NVIDIA.

    ROCm torch also reports device type "cuda" and returns the gfx version as
    the capability, e.g. (9, 0) on gfx90a, so the build must be checked too.
    """
    torch = _cuda_torch()
    if torch is None or torch.version.hip:
        return None
    return tuple(torch.cuda.get_device_capability(device))


def is_nvidia_sm90() -> bool:
    return nvidia_capability() == (9, 0)


def has_nvidia_sm90_devices(count: int) -> bool:
    torch = _cuda_torch()
    if torch is None or torch.version.hip or torch.cuda.device_count() < count:
        return False
    return all(nvidia_capability(device) == (9, 0) for device in range(count))


def require_nvidia_sm90():
    if not is_nvidia_sm90():
        pytest.skip(NVIDIA_SM90_REASON)


requires_nvidia_sm90 = pytest.mark.skipif(not is_nvidia_sm90(), reason=NVIDIA_SM90_REASON)
