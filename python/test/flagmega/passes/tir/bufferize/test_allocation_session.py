# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega.ir.bufferization import AllocationStrategy, MemorySpace
from triton.flagmega.passes.tir.bufferize import BufferLifetime, BufferizationOptions
from triton.flagmega.passes.tir.bufferize.allocation_session import AllocationSession


@pytest.mark.parametrize("level", ["fast", "optimized"])
def test_isomorphic_physical_problem_reuses_result_with_current_ids(level):
    space = MemorySpace("workspace", "device", 64, 4096, AllocationStrategy.REUSE)
    session = AllocationSession(BufferizationOptions((space, ), optimization_level=level))
    original = (BufferLifetime("a", 64, 64, 0, 1), BufferLifetime("b", 128, 64, 2, 3))
    first = session.allocate(original, space)
    renamed = tuple(replace(value, id="clone." + value.id) for value in original)
    second = session.allocate(renamed, space)
    assert second.offset_map == {"clone." + key: value for key, value in first.offsets}
    assert (session.hits, session.misses) == (1, 1)
    session.allocate(renamed, space, avoid_reuse=(("clone.a", "clone.b"), ))
    assert session.misses == 2
    session.allocate((replace(renamed[0], live_end=2), renamed[1]), space)
    session.allocate((replace(renamed[0], alignment=256), renamed[1]), space)
    session.allocate((replace(renamed[0], nbytes=128), renamed[1]), space)
    session.allocate(renamed, replace(space, maximum_bytes=8192))
    assert session.misses == 6
    fresh = AllocationSession(BufferizationOptions((space, ), optimization_level=level))
    fresh.allocate(original, space)
    assert (fresh.hits, fresh.misses) == (0, 1)
