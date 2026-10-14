# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nvidia-sm89 pins an exact Ada (8, 9) capability, not any 8.x GPU.

The Sm89Capability resource model (Ada opt-in smem / register file) does not
describe other 8.x parts such as A100 (sm_80), so the device check must match
(8, 9) exactly rather than the compute-capability major alone.
"""

import pytest

from triton.flagmega.errors import RuntimeContractError
from triton.flagmega.runtime.module import _validate_artifact_device


class _FakeCuda:
    def __init__(self, capability):
        self._capability = capability

    def is_available(self):
        return True

    def get_device_capability(self, _device):
        return self._capability


class _FakeTorch:
    def __init__(self, capability):
        self.cuda = _FakeCuda(capability)

    def device(self, name):
        return name


def test_nvidia_sm89_accepts_exact_ada():
    _validate_artifact_device(_FakeTorch((8, 9)), "cuda", "nvidia-sm89")


@pytest.mark.parametrize("capability", [(8, 0), (8, 6), (8, 7), (7, 5), (9, 0)])
def test_nvidia_sm89_rejects_non_ada(capability):
    with pytest.raises(RuntimeContractError):
        _validate_artifact_device(_FakeTorch(capability), "cuda", "nvidia-sm89")


def test_nvidia_sm90_remains_exact():
    _validate_artifact_device(_FakeTorch((9, 0)), "cuda", "nvidia-sm90")
    with pytest.raises(RuntimeContractError):
        _validate_artifact_device(_FakeTorch((9, 1)), "cuda", "nvidia-sm90")
