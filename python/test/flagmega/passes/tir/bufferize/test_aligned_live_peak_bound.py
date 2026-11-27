# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Aligned simultaneous roots have an exact elementary high-water bound."""

import pytest
from itertools import product
from triton.flagmega.ir.bufferization import AllocationStrategy, MemorySpace
from triton.flagmega.passes.tir.bufferize import BufferLifetime, SATBufferAllocator


@pytest.mark.parametrize("count", (24, 40, 60))
def test_small_aligned_roots_do_not_spend_the_search_budget_proving_padding(count):
    granularity = 256
    space = MemorySpace("workspace", "device", granularity, 4 * 1024**2, AllocationStrategy.SAT)
    lifetimes = (BufferLifetime("large", 1024**2, granularity, 0, 0),) + tuple(
        BufferLifetime(f"small{i}", 16, granularity, 0, 0) for i in range(count)
    )
    result = SATBufferAllocator(maximum_time_seconds=0.5).allocate(lifetimes, space)
    # Every start is page aligned and all roots are live together. Each
    # small root consumes one page; only the last root's trailing slack
    # can lie beyond the raw high-water objective.
    expected_raw = 1024**2 + count * granularity - (granularity - 16)
    assert result.pool_bytes == 1024**2 + count * granularity
    assert result.objectives[0].value == expected_raw
    assert result.objectives[0].status == "OPTIMAL", result.objectives


REAL_ROOTS = (
    (
        (16, 256),
        (64, 256),
        (16384, 256),
        (816, 256),
        (8192, 256),
        (16384, 256),
        (16384, 256),
        (16384, 256),
        (128, 256),
        (256, 256),
        (256, 256),
        (16384, 256),
        (16384, 256),
        (16384, 256),
        (16384, 32),
        (16384, 256),
        (16384, 32),
        (16384, 32),
        (8192, 256),
        (8192, 256),
        (65536, 256),
        (16, 256),
        (64, 256),
        (16384, 256),
        (816, 256),
        (8192, 256),
        (1152, 256),
        (32, 256),
        (32, 256),
        (65536, 256),
        (32768, 64),
        (32768, 256),
        (1024, 256),
        (524288, 256),
        (4096, 8),
        (65536, 256),
        (65536, 256),
        (128, 256),
        (32768, 256),
        (8192, 64),
        (8192, 256),
    ),
    (
        (16, 256),
        (64, 256),
        (16384, 256),
        (816, 256),
        (8192, 256),
        (3072, 256),
        (32768, 64),
        (1024, 256),
        (8192, 16),
        (8192, 256),
        (256, 256),
        (256, 256),
        (64, 256),
        (65536, 256),
        (65536, 256),
        (32768, 256),
        (8192, 256),
        (8192, 256),
        (65536, 256),
        (16, 256),
        (64, 256),
        (16384, 256),
        (816, 256),
        (8192, 256),
        (1152, 256),
        (32, 256),
        (32, 256),
        (65536, 256),
        (32768, 64),
        (32768, 256),
        (1024, 256),
        (524288, 256),
        (4096, 8),
        (65536, 256),
        (65536, 256),
        (128, 256),
        (32768, 256),
        (8192, 64),
        (8192, 256),
    ),
)


@pytest.mark.parametrize("roots", REAL_ROOTS)
def test_real_decode_roots_prove_their_exact_aligned_peak(roots):
    space = MemorySpace("workspace", "device", 256, 4 * 1024**2, AllocationStrategy.SAT)
    lifetimes = tuple(BufferLifetime(str(i), size, alignment, 0, 0) for i, (size, alignment) in enumerate(roots))
    expected_raw = sum((size + 255) // 256 * 256 for size, _ in roots) - max((-size) % 256 for size, _ in roots)
    result = SATBufferAllocator(maximum_time_seconds=0.5).allocate(lifetimes, space)
    assert result.objectives[0].status == "OPTIMAL", result.objectives
    assert result.objectives[0].value == expected_raw


@pytest.mark.parametrize("granularity", (1, 2, 4))
@pytest.mark.parametrize(
    "windows",
    (
        ((0, 0), (0, 0), (0, 0)),
        ((0, 1), (1, 2), (2, 3)),
        ((0, 0), (1, 1), (2, 2)),
        ((0, 3), (0, 1), (2, 3)),
    ),
)
def test_mixed_alignment_bound_preserves_the_exact_raw_objective(granularity, windows):
    from triton.flagmega.passes.tir.bufferize.sat_allocator import _aligned_live_peak_lower_bound

    for sizes in ((1, 2, 3), (3, 5, 2), (4, 4, 1)):
        alignments = (granularity, granularity * 2, granularity)
        lifetimes = tuple(
            BufferLifetime(str(i), size, alignment, *window)
            for i, (size, alignment, window) in enumerate(zip(sizes, alignments, windows))
        )
        upper = sum((size + alignment - 1) // alignment * alignment for size, alignment in zip(sizes, alignments))
        feasible = []
        for offsets in product(*(range(0, upper, a) for a in alignments)):
            overlaps = any(
                windows[i][0] <= windows[j][1]
                and windows[j][0] <= windows[i][1]
                and offsets[i] < offsets[j] + sizes[j]
                and offsets[j] < offsets[i] + sizes[i]
                for i in range(3)
                for j in range(i)
            )
            if not overlaps:
                feasible.append(max(offset + size for offset, size in zip(offsets, sizes)))
        exact = min(feasible)
        assert _aligned_live_peak_lower_bound(lifetimes, granularity) <= exact
        space = MemorySpace("workspace", "device", granularity, upper * 2, AllocationStrategy.SAT)
        result = SATBufferAllocator().allocate(lifetimes, space)
        assert result.objectives[0].value == exact
