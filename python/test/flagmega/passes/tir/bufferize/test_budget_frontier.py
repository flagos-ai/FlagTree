# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Keep feasible memory schedules when spending more bytes exposes new hazards."""
from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.passes.tir.bufferize import BufferPlanner, plan_buffers, plan_memory_synchronization

from python.test.flagmega.passes.tir.bufferize.test_barrier_aware_placement import _case


def _nested_case():
    _, options = _case()
    mesh = fm.Placement((2, 4), "yx", "bb")
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")

    def typ(size=16, partial=False):
        value = fm.tensor_type("float32", (size,))
        return fm.DistributedType(value, (fm.SBP.broadcast(),), mesh, fm.SBP.partial((0, 1))) if partial else value

    def kernel(name, inputs, size=16, partial=False):
        return builder.call("tir.kernel", inputs, typ(size, partial), id=name, attrs={
            "semantic_op": "test.opaque_memory_kernel", "candidate": "test.opaque_memory_kernel",
            "parameters": {}, "facts": {}, "semantic_attrs": {},
        })

    source = builder.var("workersource", typ(partial=True), id="workersource")
    other = builder.var("workerother", typ(), id="workerother")
    left = kernel("worker0", (source,), 32)
    right = kernel("worker5", (other,), 64)
    output = kernel("worker11", (left, right))
    builder.function("worker", (source, other), (output,))
    source = builder.var("mainsource", typ(partial=True), id="mainsource")
    other = builder.var("mainother", typ(), id="mainother")
    first = kernel("main0", (source, other), partial=True)
    wide = kernel("main1", (source, first), 256)
    retained = kernel("main2", (wide, first), 128)
    call = builder.call("builtin.call", (source, other), typ(), id="main3", attrs={"callee": "worker"})
    later = kernel("main4", (other,), partial=True)
    expanded = kernel("main6", (later,), 256)
    split = kernel("main7", (expanded,), 128, True)
    joined = kernel("main10", (call, split, wide), partial=True)
    output = kernel("main11", (retained, wide, joined))
    builder.function("main", (source, other), (output,))
    # This fixture models the cooperative TLE grid protocol used by SM90;
    # its grid rendezvous includes both CTA arrival and acquire phases.
    module = replace(builder.build(entry="main"), metadata={
        "memory_synchronization_protocol": {
            "schema": "flagmega.memory-synchronization-protocol/v1",
            "identity": "nvidia-sm90/tle-grid/v1",
            "grid_cta_barriers": 2, "device_grid_barriers": 2,
        },
    })
    return fm.verify_module(module), options


def _grids(module, plan):
    return sum(event.scope == "grid" for event in plan_memory_synchronization(module, plan).events)


@pytest.mark.parametrize("rounds", (1, 6))
def test_positive_budget_keeps_the_zero_spend_improvement(rounds):
    module, options = _nested_case()
    anchor = BufferPlanner(module, options).run()
    zero = plan_buffers(module, options=options)
    assert _grids(module, zero) < _grids(module, anchor)
    budget = 512
    selected = plan_buffers(module, options=replace(
        options, barrier_bytes_budget=budget, barrier_fixpoint_rounds=rounds))
    # The larger local SAT objective exposes a new cut. Its rejection must
    # retain the already feasible zero-spend improvement, not the raw layout.
    assert _grids(module, selected) <= _grids(module, zero)
    if rounds > 1:
        assert _grids(module, selected) < _grids(module, zero)
    for function in anchor.functions:
        for name, pool in function.memory_pool_map.items():
            assert selected.function_map[function.name].memory_pool_map[name].scope_bytes <= pool.scope_bytes + budget
