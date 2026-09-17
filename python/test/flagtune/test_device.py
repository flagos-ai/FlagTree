"""Validate strict backend-neutral FlagTune device discovery."""

from types import SimpleNamespace

import pytest

from triton.flagtune.runtime.device import (
    DeviceProbeError,
    UnsupportedFlagTuneDeviceError,
    probe_flagtune_device,
)


class _FakeInterface:

    def __init__(self, names):
        self._names = names

    @staticmethod
    def current_device():
        return 0

    def get_device_name(self, index):
        return self._names[index]


class _FakeActive:

    def __init__(self, backend, arch, names=("Test GPU", )):
        self._target = SimpleNamespace(backend=backend, arch=arch)
        self._interface = _FakeInterface(names)

    def get_current_target(self):
        return self._target

    def get_device_interface(self):
        return self._interface


class _FakePPUActive(_FakeActive):
    """Model the active driver class exposed by FlagTree's PPU backend."""


_FakePPUActive.__module__ = "triton.backends.ppu.driver"


class _FakeHCUActive(_FakeActive):
    """Model the active driver class exposed by FlagTree's Hygon backend."""


_FakeHCUActive.__module__ = "triton.backends.hcu.driver"


def test_probe_cuda_uses_native_architecture_and_explicit_device(monkeypatch):
    from triton.flagtune.runtime import device

    monkeypatch.setattr(
        device,
        "_active_driver",
        lambda: _FakeActive("cuda", 90, ("NVIDIA H20", "NVIDIA H20")),
    )
    descriptor = probe_flagtune_device(1)

    assert descriptor.backend == "cuda"
    assert descriptor.vendor == "nvidia"
    assert descriptor.torch_device_type == "cuda"
    assert descriptor.device_name == "NVIDIA H20"
    assert descriptor.architecture == "sm90"
    assert descriptor.device_index == 1


def test_probe_hip_preserves_gfx_architecture(monkeypatch):
    from triton.flagtune.runtime import device

    monkeypatch.setattr(
        device,
        "_active_driver",
        lambda: _FakeActive("hip", "gfx942", ("AMD Instinct MI300X", )),
    )
    descriptor = probe_flagtune_device()

    assert descriptor.backend == "hip"
    assert descriptor.vendor == "amd"
    assert descriptor.torch_device_type == "cuda"
    assert descriptor.architecture == "gfx942"


def test_probe_maca_uses_metax_runtime_contract(monkeypatch):
    from triton.flagtune.runtime import device

    monkeypatch.setattr(
        device,
        "_active_driver",
        lambda: _FakeActive("maca", 80, ("MetaX C550", )),
    )
    descriptor = probe_flagtune_device()

    assert descriptor.backend == "maca"
    assert descriptor.vendor == "metax"
    assert descriptor.torch_device_type == "cuda"
    assert descriptor.device_name == "MetaX C550"
    assert descriptor.architecture == "sm80"


def test_probe_cuda_target_from_ppu_driver_uses_thead_runtime_contract(monkeypatch):
    from triton.flagtune.runtime import device

    monkeypatch.setattr(
        device,
        "_active_driver",
        lambda: _FakePPUActive("cuda", 80, ("PPU-ZW810E", )),
    )
    descriptor = probe_flagtune_device()

    assert descriptor.backend == "cuda"
    assert descriptor.vendor == "thead"
    assert descriptor.torch_device_type == "cuda"
    assert descriptor.device_name == "PPU-ZW810E"
    assert descriptor.architecture == "sm80"


def test_probe_musa_uses_mthreads_runtime_contract(monkeypatch):
    from triton.flagtune.runtime import device

    monkeypatch.setattr(
        device,
        "_active_driver",
        lambda: _FakeActive("musa", 31, ("MTT S5000", )),
    )
    descriptor = probe_flagtune_device()

    assert descriptor.backend == "musa"
    assert descriptor.vendor == "mthreads"
    assert descriptor.torch_device_type == "musa"
    assert descriptor.device_name == "MTT S5000"
    assert descriptor.architecture == "musa31"


def test_probe_hip_target_from_hcu_driver_uses_hygon_runtime_contract(monkeypatch):
    from triton.flagtune.runtime import device

    monkeypatch.setattr(
        device,
        "_active_driver",
        lambda: _FakeHCUActive("hip", "gfx936", ("BW", )),
    )
    descriptor = probe_flagtune_device()

    assert descriptor.backend == "hip"
    assert descriptor.vendor == "hygon"
    assert descriptor.torch_device_type == "cuda"
    assert descriptor.device_name == "BW"
    assert descriptor.architecture == "gfx936"


def test_probe_unknown_backend_fails_at_device_boundary(monkeypatch):
    from triton.flagtune.runtime import device

    monkeypatch.setattr(device, "_active_driver", lambda: _FakeActive("xpu", "pvc"))
    with pytest.raises(
            UnsupportedFlagTuneDeviceError,
            match="does not support Triton backend 'xpu'.*cuda, hip, maca, musa",
    ):
        probe_flagtune_device()


def test_probe_rejects_invalid_native_architecture(monkeypatch):
    from triton.flagtune.runtime import device

    monkeypatch.setattr(device, "_active_driver", lambda: _FakeActive("cuda", "hopper"))
    with pytest.raises(DeviceProbeError, match="invalid architecture"):
        probe_flagtune_device()
