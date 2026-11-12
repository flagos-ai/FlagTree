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


@pytest.mark.parametrize("arch", ["Ascend910B1", "Ascend910_9392", "Ascend310P3", "Ascend910B4-1"])
def test_probe_npu_preserves_ascend_soc_and_device_index(monkeypatch, arch):
    from triton.flagtune.runtime import device
    from triton.flagtune.contract.identity import make_platform_key

    active = _FakeActive("npu", arch, ("Ascend910B", "Huawei Ascend910B"))
    monkeypatch.setattr(device, "_active_driver", lambda: active)
    descriptor = probe_flagtune_device(1)

    assert descriptor.backend == "npu"
    assert descriptor.vendor == "huawei"
    assert descriptor.torch_device_type == "npu"
    assert descriptor.architecture == arch.lower()
    assert descriptor.device_index == 1
    assert make_platform_key(descriptor.vendor, descriptor.device_name) == "huawei-ascend910"


@pytest.mark.parametrize("arch", ["", "sm90", "gfx942", "Ascend", "Ascend910/../B"])
def test_probe_npu_rejects_invalid_soc(monkeypatch, arch):
    from triton.flagtune.runtime import device

    monkeypatch.setattr(device, "_active_driver", lambda: _FakeActive("npu", arch))
    with pytest.raises(DeviceProbeError, match="invalid architecture"):
        probe_flagtune_device()


@pytest.mark.parametrize("backend", ["cuda", "hip", "maca", "musa", "xpu", "ascend"])
def test_only_npu_backend_is_supported(monkeypatch, backend):
    from triton.flagtune.runtime import device
    monkeypatch.setattr(device, "_active_driver", lambda: _FakeActive(backend, "Ascend910B4-1"))
    assert device.registered_device_backends() == ("npu", )
    with pytest.raises(UnsupportedFlagTuneDeviceError, match="registered backends: npu"):
        probe_flagtune_device()
