# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.ir.bufferization import (
    AllocationPolicy,
    AllocationStrategy,
    MemorySharingScope,
)
from triton.flagmega.targets import Sm90Capability
from triton.flagmega.targets.nvidia import sm90_bufferization_options


def test_sm90_memory_spaces_derive_shared_capacity_from_machine():
    options = sm90_bufferization_options(
        Sm90Capability(max_shared_memory_bytes=196608)
    )
    spaces = options.memory_space_map

    assert tuple(spaces) == (
        "workspace", "block_local_data", "rdata", "shared", "external"
    )
    assert spaces["workspace"].strategy is AllocationStrategy.SAT
    assert spaces["block_local_data"].strategy is AllocationStrategy.SAT
    assert spaces["block_local_data"].sharing_scope is MemorySharingScope.BLOCK
    assert options.block_local == "block_local_data"
    assert spaces["rdata"].strategy is AllocationStrategy.LINEAR
    assert spaces["shared"].strategy is AllocationStrategy.SAT
    assert (
        spaces["shared"].allocation_policy
        is AllocationPolicy.GRANULARITY_ALIGNED
    )
    assert spaces["shared"].maximum_bytes == 196608
    assert spaces["shared"].allocation_bytes(8704) == 8704
    # The QKV MMA pipeline uses a 2-stage 32x1024 BF16 RHS plus one
    # complete-K 1x2048 BF16 LHS.  CUDA Shared is byte-addressable and the
    # nncase target contract aligns this combined SAT arena; rounding the
    # entire 135168-byte arena to 262144 would reject a legal launch.
    assert spaces["shared"].allocation_bytes(135168) == 135168
    assert spaces["external"].strategy is AllocationStrategy.EXTERNAL
