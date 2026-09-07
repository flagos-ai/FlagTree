# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Assembler occupancy assumptions must agree with the residency contract."""

import pytest

from triton.flagmega.runtime.nvidia_launch import compilation_options
from triton.flagmega.runtime.prepared import ResourceContract


@pytest.mark.parametrize("warps,blocks", [(4, 1), (8, 1), (4, 2), (4, 4)])
def test_compilation_uses_contract_launch_bounds(warps, blocks):
    contract = ResourceContract(compute_num_warps=warps, resident_blocks_per_sm=blocks)
    assert compilation_options(contract) == {
        "num_warps": warps,
        "ptx_options": f"--minnctapersm={blocks}",
    }
    assert contract.forbid_spills
