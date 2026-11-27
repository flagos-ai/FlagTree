# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.bufferization import (
    AllocationPolicy,
    AllocationStrategy,
    MemoryAllocationScope,
    MemorySharingScope,
    MemorySpace,
)


def test_memory_space_separates_arena_ownership_from_physical_sharing():
    space = MemorySpace(
        "data",
        "device",
        256,
        1 << 20,
        AllocationStrategy.SAT,
        AllocationPolicy.GRANULARITY_ALIGNED,
        MemoryAllocationScope.FUNCTION,
        MemorySharingScope.BLOCK,
    )

    assert space.allocation_scope is MemoryAllocationScope.FUNCTION
    assert space.sharing_scope is MemorySharingScope.BLOCK
    assert space.shared_scope == "function"
    assert MemorySpace.from_data(space.to_data()) == space


def test_legacy_memory_space_decodes_ownership_and_conservative_visibility():
    legacy = {
        "name": "workspace",
        "kind": "device",
        "granularity": 256,
        "maximum_bytes": 4096,
        "strategy": "sat",
        "allocation_policy": "granularity_aligned",
        "shared_scope": "function",
    }
    workspace = MemorySpace.from_data(legacy)
    shared = MemorySpace.from_data({
        **legacy,
        "name": "shared",
        "kind": "shared",
    })

    assert workspace.allocation_scope is MemoryAllocationScope.FUNCTION
    assert workspace.sharing_scope is MemorySharingScope.CHIP
    assert shared.sharing_scope is MemorySharingScope.BLOCK


@pytest.mark.parametrize(
    ("strategy", "scope"),
    (
        (AllocationStrategy.EXTERNAL, MemoryAllocationScope.FUNCTION),
        (AllocationStrategy.SAT, MemoryAllocationScope.EXTERNAL),
    ),
)
def test_external_strategy_and_allocation_scope_must_agree(strategy, scope):
    with pytest.raises(IRSchemaError, match="external"):
        MemorySpace(
            "invalid",
            "device",
            1,
            4096,
            strategy,
            allocation_scope=scope,
        )
