# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""An amd-* artifact validates the device's gfx arch, not just a HIP build.

gfx1100 (RDNA3) and gfx1201 (RDNA4) differ in ISA / WMMA, so loading a
gfx1201-built artifact on a gfx1100 device (or vice versa) must be rejected at
load time instead of failing later at JIT / launch.
"""

import pytest

from triton.flagmega.errors import RuntimeContractError
from triton.flagmega.runtime.module import _validate_artifact_device


class _Props:
    def __init__(self, arch):
        self.gcnArchName = arch


class _FakeCuda:
    def __init__(self, arch):
        self._arch = arch

    def is_available(self):
        return True

    def get_device_properties(self, _device):
        return _Props(self._arch)


class _FakeVersion:
    def __init__(self, hip):
        self.hip = hip


class _FakeTorch:
    def __init__(self, arch, hip="7.2"):
        self.cuda = _FakeCuda(arch)
        self.version = _FakeVersion(hip)

    def device(self, name):
        return name


def test_amd_gfx1100_accepts_matching_device():
    _validate_artifact_device(_FakeTorch("gfx1100"), "cuda", "amd-gfx1100")


def test_amd_gfx1201_accepts_feature_suffixed_arch():
    _validate_artifact_device(_FakeTorch("gfx1201:sramecc+:xnack-"), "cuda", "amd-gfx1201")


@pytest.mark.parametrize(
    "arch,target",
    [("gfx1100", "amd-gfx1201"), ("gfx1201", "amd-gfx1100"), ("", "amd-gfx1100")],
)
def test_amd_rejects_arch_mismatch(arch, target):
    with pytest.raises(RuntimeContractError):
        _validate_artifact_device(_FakeTorch(arch), "cuda", target)


def test_amd_requires_hip_build():
    with pytest.raises(RuntimeContractError):
        _validate_artifact_device(_FakeTorch("gfx1100", hip=None), "cuda", "amd-gfx1100")
