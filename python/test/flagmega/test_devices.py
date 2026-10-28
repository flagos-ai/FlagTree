# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest

from python.test.flagmega.devices import (executable_target, has_nvidia_sm90_devices, is_nvidia_sm90,
                                          nvidia_capability, require_nvidia_sm90)

torch = pytest.importorskip("torch")


def _device(monkeypatch, *, available, hip, capability, arch="", count=1, current=0):
    """``capability`` is one tuple for every device, or a per-device dict."""
    capabilities = capability if isinstance(capability, dict) else {}

    def get_capability(device=None):
        device = current if device is None else device
        return capabilities.get(device, capability) if capabilities else capability

    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", get_capability)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count if available else 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda *args: SimpleNamespace(gcnArchName=arch))


@pytest.mark.parametrize("available,hip,capability,expected",
                         [(True, None, (9, 0), True),  # NVIDIA H100 / H800
                          (True, "6.2", (9, 0), False),  # ROCm gfx90a (MI200) also reports (9, 0)
                          (True, "7.2", (12, 0), False),  # ROCm gfx1201 (R9700)
                          (True, None, (8, 9), False),  # NVIDIA Ada
                          (True, None, (12, 0), False),  # NVIDIA Blackwell consumer
                          (False, None, (9, 0), False),  # no GPU
                          ])
def test_only_nvidia_sm90_runs_sm90_artifacts(monkeypatch, available, hip, capability, expected):
    _device(monkeypatch, available=available, hip=hip, capability=capability)
    assert is_nvidia_sm90() is expected


def test_require_nvidia_sm90_skips_on_rocm_gfx90a(monkeypatch):
    _device(monkeypatch, available=True, hip="6.2", capability=(9, 0))
    with pytest.raises(pytest.skip.Exception):
        require_nvidia_sm90()


@pytest.mark.parametrize("available,hip,capability,arch,expected", [
    (True, None, (9, 0), "", "nvidia-sm90"),
    (True, None, (12, 0), "", "nvidia-sm120"),
    (True, None, (8, 9), "", None),
    (True, "7.2", (12, 0), "gfx1201", "amd-gfx1201"),  # R9700 also reports (12, 0)
    (True, "7.2", (11, 0), "gfx1100:sramecc-:xnack-", "amd-gfx1100"),
    (True, "6.2", (9, 0), "gfx90a:sramecc+:xnack-", None),  # no FlagMega target, never sm90
    (True, "7.2", (9, 5), "gfx950", None),
    (False, None, (9, 0), "", None),
])
def test_executable_target_never_maps_rocm_to_nvidia(monkeypatch, available, hip, capability, arch, expected):
    _device(monkeypatch, available=available, hip=hip, capability=capability, arch=arch)
    assert executable_target() == expected


def test_nvidia_capability_is_none_on_rocm(monkeypatch):
    _device(monkeypatch, available=True, hip="7.2", capability=(12, 0), arch="gfx1201")
    assert nvidia_capability() is None
    _device(monkeypatch, available=True, hip=None, capability=(12, 0))
    assert nvidia_capability() == (12, 0)


@pytest.mark.parametrize("hip,capability,count,expected", [
    (None, (9, 0), 4, True),
    (None, (9, 0), 3, False),
    ("6.2", (9, 0), 4, False),  # four MI200 GPUs
    (None, (12, 0), 4, False),
])
def test_multi_gpu_sm90_guard(monkeypatch, hip, capability, count, expected):
    _device(monkeypatch, available=True, hip=hip, capability=capability, count=count)
    assert has_nvidia_sm90_devices(4) is expected


def test_sm90_guards_follow_the_current_device(monkeypatch):
    # Mixed machine: GPU 0 is Ada, GPU 1 is an H800. Tests that allocate on
    # device="cuda" run on the current device, so the guard must query it.
    mixed = {0: (8, 9), 1: (9, 0)}
    _device(monkeypatch, available=True, hip=None, capability=mixed, count=2, current=1)
    assert nvidia_capability() == (9, 0)
    assert is_nvidia_sm90()
    require_nvidia_sm90()
    _device(monkeypatch, available=True, hip=None, capability=mixed, count=2, current=0)
    assert nvidia_capability() == (8, 9)
    assert not is_nvidia_sm90()
    with pytest.raises(pytest.skip.Exception):
        require_nvidia_sm90()


def test_explicit_device_queries_keep_their_device(monkeypatch):
    # executable_target() selects for the runtime it loads on cuda:0, and the
    # multi-GPU guard checks the first N devices, whatever the current device.
    mixed = {0: (8, 9), 1: (9, 0)}
    _device(monkeypatch, available=True, hip=None, capability=mixed, count=2, current=1)
    assert executable_target() is None
    assert nvidia_capability(0) == (8, 9)
    assert not has_nvidia_sm90_devices(2)
    _device(monkeypatch, available=True, hip=None, capability={0: (9, 0), 1: (8, 9)}, count=2, current=1)
    assert executable_target() == "nvidia-sm90"
