# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from python.test.flagmega.devices import is_nvidia_sm90, require_nvidia_sm90

torch = pytest.importorskip("torch")


def _device(monkeypatch, *, available, hip, capability):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: capability)


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
