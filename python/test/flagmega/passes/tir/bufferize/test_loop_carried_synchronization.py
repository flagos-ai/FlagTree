# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A loop body re-runs, so its own last write races its own first read."""

from triton.flagmega import ir as fm
from triton.flagmega.ir import DType, IRBuilder, make_buffer_plan, tensor_type
from triton.flagmega.passes.tir.bufferize import plan_memory_synchronization


def carried_loop_module(trips=4):
    """Build a loop whose body reads, then overwrites, the carried value.

    The body's first kernel reads the carry and its last kernel writes it, so
    trip N's write is ordered against trip N+1's read only if the analysis
    treats the body as cyclic.
    """
    builder = IRBuilder(dialect="tir", stage="selected_tir")
    value_type = tensor_type(DType.FLOAT32, [1, 128])
    index_type = tensor_type(DType.INT64, [])

    iv = builder.var("iv", index_type, id="iv")
    carried = builder.var("carried", value_type, id="carried")
    consumed = builder.call("tir.kernel", [carried], value_type, id="body_read")
    produced = builder.call("tir.kernel", [consumed], value_type, id="body_write")
    builder.region("body", (iv, carried), (produced,), iter_count=1)

    lower = builder.var("lower", index_type, id="lower")
    upper = builder.var("upper", index_type, id="upper")
    step = builder.var("step", index_type, id="step")
    seed = builder.var("seed", value_type, id="seed")
    loop = builder.call(
        "control.for", [lower, upper, step, seed],
        fm.TupleType((value_type,)), id="loop",
        attrs={"callee": "body", "iter_count": 1},
    )
    output = builder.call("builtin.get_item", [loop], value_type, id="output",
                          attrs={"index": 0})
    builder.function("main", (lower, upper, step, seed), (output,))
    return builder.build(entry="main")


def test_loop_body_orders_its_own_carried_write_against_the_next_trip():
    module = carried_loop_module()
    plan = make_buffer_plan(module)
    events = plan_memory_synchronization(module, plan).events
    wrap = [
        event for event in events
        if event.function == "body"
        and event.after == "body_write"
        and event.before == "body_read"
    ]
    assert wrap, (
        "The body's trailing write to the carry must be ordered against the "
        f"next trip's read of it; planned events were {events!r}"
    )
    # The pair spans the trip boundary either way round: the trailing write
    # must not land before the leading read of the same bytes, and the
    # leading read must see the previous trip's write.
    assert set(wrap[0].hazards) & {"WRITE->READ", "READ->WRITE"}
