# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.tir.bufferize import BufferizationOptions, plan_memory_synchronization


@pytest.mark.parametrize("intermediate_axes", [(0, 1), (0,)])
def test_narrow_barrier_does_not_forget_a_pending_cross_owner_write(intermediate_axes):
    placement = fm.Placement((2, 4), "yx", "bb")
    tensor = fm.tensor_type("float32", (1, 128))

    def distributed(axes):
        return fm.DistributedType(
            tensor,
            (fm.SBP.broadcast(), fm.SBP.split_contiguous(axes) if axes else fm.SBP.broadcast()),
            placement,
        )

    all_split = distributed((0, 1))
    intermediate = distributed(intermediate_axes)
    broadcast = distributed(())
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    source = builder.var("source", all_split, id="source")
    produced = builder.call("tir.kernel", (source,), all_split, id="produced")
    first_input = produced if intermediate == all_split else builder.call(
        "distributed.sharded_view", (produced,), intermediate, id="first_view", attrs={"new_type": intermediate}
    )
    first_reader = builder.call("tir.kernel", (first_input,), intermediate, id="first_reader")
    full_view = builder.call(
        "distributed.sharded_view", (produced,), broadcast, id="full_view", attrs={"new_type": broadcast}
    )
    last_reader = builder.call("tir.kernel", (full_view,), broadcast, id="last_reader")
    builder.function("main", (source,), (first_reader, last_reader))
    module = builder.build(entry="main")
    options = BufferizationOptions.generic()
    options = replace(options, block_local=None, memory_spaces=tuple(
        replace(space, sharing_scope=fm.MemorySharingScope.CHIP)
        if space.name == "workspace" else space for space in options.memory_spaces
    ))
    plan = fm.make_buffer_plan(module, options=options)

    synchronization = plan_memory_synchronization(module, plan)

    by_before = {event.before: event for event in synchronization.events}
    early = by_before["first_reader"]
    assert (early.scope, early.axis_group_axes) == (
        ("block", ()) if len(intermediate_axes) == 2 else ("grid", (1,))
    )
    assert "last_reader" in by_before
    assert by_before["last_reader"].scope == "grid"
    assert by_before["last_reader"].axis_group_axes == ()
    assert "WRITE->READ" in by_before["last_reader"].hazards
