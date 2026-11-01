# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""New byte hazards exposed by a SAT placement remain in the next round."""

from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.passes.tir.bufferize import (
    BufferPlanner,
    plan_buffers,
    plan_memory_synchronization,
)

from python.test.flagmega.passes.tir.bufferize.test_barrier_aware_placement import _case


def _exposed_reuse_pair():
    _, options = _case()
    mesh = fm.Placement((2, 4), "yx", "bb")

    def typ(size, partial=False):
        value = fm.tensor_type("float32", (size,))
        return fm.DistributedType(
            value, (fm.SBP.broadcast(),), mesh, fm.SBP.partial((0, 1))
        ) if partial else value

    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")

    def kernel(name, inputs, size, partial=False):
        # Opaque selected kernels declare their result and ordinary read/write
        # footprint. The planner need not know the arithmetic to place them.
        return builder.call("tir.kernel", inputs, typ(size, partial), id=name, attrs={
            "semantic_op": "test.opaque_memory_kernel", "candidate": "test.opaque_memory_kernel",
            "parameters": {}, "facts": {}, "semantic_attrs": {},
        })

    source = builder.var("source", typ(16, True), id="source")
    other = builder.var("other", typ(16), id="other")
    anchor = kernel("anchor", (other,), 1024)
    anchor_done = kernel("anchor_done", (anchor,), 16)
    early = kernel("early", (other,), 16)
    retained = kernel("retained", (other, source), 16, True)
    first_output = kernel("first_output", (other,), 16, True)
    middle = kernel("middle", (anchor_done,), 32, True)
    later = kernel("later", (middle, early), 16, True)
    projected = kernel("projected", (retained,), 128)
    second_output = kernel("second_output", (projected,), 128, True)
    tail = kernel("tail", (retained,), 16, True)
    final_output = kernel("final_output", (tail, later), 128, True)
    builder.function("main", (source, other), (first_output, second_output, final_output))
    return fm.verify_module(builder.build(entry="main")), options


@pytest.mark.parametrize("rounds", (2, 8))
def test_new_reuse_pair_in_the_same_pool_reaches_the_next_round(rounds):
    module, options = _exposed_reuse_pair()
    baseline = BufferPlanner(module, options).run()
    once = plan_buffers(module, options=options)
    repeated = plan_buffers(module, options=replace(options, barrier_fixpoint_rounds=rounds))
    before = plan_memory_synchronization(module, once)
    after = plan_memory_synchronization(module, repeated)
    # The first relocation removes one cut but exposes a different pair in
    # the same (function, space) bucket. The next placement removes that cut.
    assert len(after.events) < len(before.events)
    assert repeated.workspace_bytes == once.workspace_bytes == baseline.workspace_bytes


def test_unchanged_pair_set_stops_without_another_full_planning_round(monkeypatch):
    from triton.flagmega.passes.tir.bufferize import planner

    module, options = _case()
    module = fm.verify_module(replace(module, nodes=tuple(
        replace(node, attrs={
            "semantic_op": "test.opaque_memory_kernel", "candidate": "test.opaque_memory_kernel",
            "parameters": {}, "facts": {}, "semantic_attrs": {},
        }) if node.op == "tir.kernel" else node for node in module.nodes
    )))
    planned = []
    original = planner.BufferPlanner.run

    def observe(self):
        result = original(self)
        key = (tuple(sorted(getattr(self, "completion_ends", {}).items())),
               self.options.barrier_bytes_budget,
               tuple((name, tuple(pairs)) for name, pairs in sorted(self.reuse_preferences.items())))
        planned.append((key, result))
        return result

    monkeypatch.setattr(planner.BufferPlanner, "run", observe)
    selected = plan_buffers(module, options=replace(options, barrier_fixpoint_rounds=8))
    # Optional lifetime proposals are separate searches. No budget/family
    # should re-plan the same problem after its pair set stops changing.
    keys = [key for key, _ in planned]
    assert len(keys) == len(set(keys))
    assert any(selected is result for _, result in planned)
