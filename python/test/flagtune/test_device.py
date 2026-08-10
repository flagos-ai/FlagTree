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


def test_probe_unknown_backend_fails_at_device_boundary(monkeypatch):
    from triton.flagtune.runtime import device

    monkeypatch.setattr(device, "_active_driver", lambda: _FakeActive("xpu", "pvc"))
    with pytest.raises(
            UnsupportedFlagTuneDeviceError,
            match="does not support Triton backend 'xpu'.*cuda, hip",
    ):
        probe_flagtune_device()


def test_probe_rejects_invalid_native_architecture(monkeypatch):
    from triton.flagtune.runtime import device

    monkeypatch.setattr(device, "_active_driver", lambda: _FakeActive("cuda", "hopper"))
    with pytest.raises(DeviceProbeError, match="invalid architecture"):
        probe_flagtune_device()
