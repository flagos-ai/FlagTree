# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nvidia-sm89 must carry Ada's shared-memory limit in its runtime contract.

Without an sm89 branch, _resource_contract falls through to the default
ResourceContract, which keeps Hopper's 227,328-byte shared-memory capacity. A
kernel using 102,400 bytes would then pass the resource check even though Ada
(sm_89) only exposes a 99 KB (101,376-byte) opt-in limit.
"""

from triton.flagmega.runtime.module import _resource_contract
from triton.flagmega.runtime.prepared import ResourceContract

_ADA_SMEM = 101376
_HOPPER_SMEM = ResourceContract.shared_memory_capacity_bytes  # dataclass default (227328)


def _contract(target):
    return _resource_contract({"num_warps": 4}, target)


def test_nvidia_sm89_uses_ada_shared_memory_limit():
    assert _contract("nvidia-sm89").shared_memory_capacity_bytes == _ADA_SMEM


def test_nvidia_sm90_keeps_the_hopper_default():
    assert _contract("nvidia-sm90").shared_memory_capacity_bytes == _HOPPER_SMEM
    assert _HOPPER_SMEM == 227328


def test_sm89_contract_rejects_a_kernel_above_ada_smem():
    contract = _contract("nvidia-sm89")
    # A kernel just over Ada's limit must not be considered to fit, while the
    # same kernel is within Hopper's larger budget.
    kernel_smem = 102400
    assert kernel_smem > contract.shared_memory_capacity_bytes
    assert kernel_smem < _HOPPER_SMEM
