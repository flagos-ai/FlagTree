# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Device predicates shared by FlagMega GPU tests."""

import pytest

NVIDIA_SM90_REASON = "requires an NVIDIA SM90 GPU"

_NVIDIA_TARGETS = {(9, 0): "nvidia-sm90", (12, 0): "nvidia-sm120"}
_AMD_TARGETS = {"gfx1100": "amd-gfx1100", "gfx1201": "amd-gfx1201"}


def _cuda_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


def nvidia_capability(device=None):
    """The NVIDIA compute capability of ``device`` (default: the current CUDA
    device, like ``torch.cuda.get_device_capability()``), or None off NVIDIA.

    ROCm torch also reports device type "cuda" and returns the gfx version as
    the capability, e.g. (9, 0) on gfx90a, so the build must be checked too.
    """
    torch = _cuda_torch()
    if torch is None or torch.version.hip:
        return None
    if device is None:
        device = torch.cuda.current_device()
    return tuple(torch.cuda.get_device_capability(device))


def is_nvidia_sm90(device=None) -> bool:
    return nvidia_capability(device) == (9, 0)


def has_nvidia_sm90_devices(count: int) -> bool:
    torch = _cuda_torch()
    if torch is None or torch.version.hip or torch.cuda.device_count() < count:
        return False
    return all(nvidia_capability(device) == (9, 0) for device in range(count))


def executable_target():
    """The FlagMega target that can run on this process's first GPU, or None."""
    torch = _cuda_torch()
    if torch is None:
        return None
    if torch.version.hip:
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        return _AMD_TARGETS.get(arch)
    return _NVIDIA_TARGETS.get(nvidia_capability(0))


def require_nvidia_sm90():
    if not is_nvidia_sm90():
        pytest.skip(NVIDIA_SM90_REASON)


requires_nvidia_sm90 = pytest.mark.skipif(not is_nvidia_sm90(), reason=NVIDIA_SM90_REASON)
