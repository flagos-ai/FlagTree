# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.bufferization import (
    AllocationStrategy,
    MemoryAllocationScope,
    MemorySharingScope,
    MemorySpace,
)
from triton.flagmega.passes.tir.bufferize import (
    BufferizationOptions,
    plan_memory_synchronization,
)


def _options(scope: MemorySharingScope) -> BufferizationOptions:
    maximum = (1 << 31) - 1
    return BufferizationOptions((
        MemorySpace(
            "workspace", "device", 64, maximum, AllocationStrategy.SAT,
            allocation_scope=MemoryAllocationScope.FUNCTION,
            sharing_scope=scope,
        ),
        MemorySpace(
            "rdata", "readonly_device", 64, maximum,
            AllocationStrategy.LINEAR,
            allocation_scope=MemoryAllocationScope.MODULE,
            sharing_scope=MemorySharingScope.CHIP,
        ),
        MemorySpace(
            "external", "external", 1, maximum,
            AllocationStrategy.EXTERNAL,
            allocation_scope=MemoryAllocationScope.EXTERNAL,
            sharing_scope=MemorySharingScope.CHIP,
        ),
    ))


def _module():
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    value_type = fm.tensor_type("float32", (16,))
    source = builder.var("source", value_type, id="source")
    produced = builder.call("tir.kernel", (source,), value_type, id="produced")
    result = builder.call("tir.kernel", (produced,), value_type, id="result")
    builder.function("main", (source,), (result,))
    return builder.build(entry="main")


@pytest.mark.parametrize(
    ("scope", "expected"),
    (
        (MemorySharingScope.BLOCK, "block"),
        (MemorySharingScope.DIE, "grid"),
        (MemorySharingScope.CHIP, "grid"),
    ),
)
def test_inferred_hazard_scope_comes_from_target_memory_space(scope, expected):
    module = _module()
    plan = fm.make_buffer_plan(module, options=_options(scope))

    event = plan_memory_synchronization(module, plan).events[0]

    assert event.scope == expected
