# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest

from python.test.flagmega.devices import (has_nvidia_sm90_devices, is_nvidia_sm90, nvidia_capability,
                                          require_nvidia_sm90)

torch = pytest.importorskip("torch")


def _device(monkeypatch, *, available, hip, capability, arch="", count=1):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: capability)
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
