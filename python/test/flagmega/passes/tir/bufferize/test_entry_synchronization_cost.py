# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Schedule costs belong to the closed entry, including its real call counts."""

from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.ir.bufferization import MemorySynchronizationPlan, SynchronizationEvent
from triton.flagmega.ir.ops.core import construction_scope
from triton.flagmega.passes.tir.bufferize import synchronization
from triton.flagmega.passes.tir.bufferize.reuse_preferences import dominates_memory_schedule


def _call_module(edges, *, entry="main"):
    builder = fm.IRBuilder(dialect="high_level", stage="packed")
    tensor = fm.tensor_type("float32", (16,))
    for name, callees in edges.items():
        parameter = builder.var(name, tensor, id=f"{name}_input")
        value = builder.call("math.silu", (parameter,), tensor, id=f"{name}_head")
        for index, callee in enumerate(callees):
            value = builder.call("builtin.call", (value,), tensor, id=f"{name}_call{index}", attrs={"callee": callee})
        value = builder.call("math.silu", (value,), tensor, id=f"{name}_tail")
        builder.function(name, (parameter,), (value,))
    return fm.verify_module(builder.build(entry=entry))


def _loop_module(*, lower=0, upper=3, step=1, body_calls=0, inner_upper=None):
    builder = fm.IRBuilder(dialect="high_level", stage="packed")
    tensor = fm.tensor_type("float32", (16,))
    index_type = fm.tensor_type("int64", ())
    if body_calls:
        parameter = builder.var("worker", tensor, id="worker_input")
        value = builder.call("math.silu", (parameter,), tensor, id="worker_head")
        value = builder.call("math.silu", (value,), tensor, id="worker_tail")
        builder.function("worker", (parameter,), (value,))

    def region(name, nested=None):
        iv = builder.var(f"{name}_iv", index_type, id=f"{name}_iv")
        carried = builder.var(f"{name}_input", tensor, id=f"{name}_input")
        value = builder.call("math.silu", (carried,), tensor, id=f"{name}_head")
        for index in range(body_calls):
            value = builder.call("builtin.call", (value,), tensor, id=f"{name}_call{index}", attrs={"callee": "worker"})
        if nested is not None:
            with construction_scope(builder):
                start = fm.F.builtin.scalar_const(index_type, 0, name=f"{name}_lower")
                stop = fm.F.builtin.scalar_const(index_type, inner_upper, name=f"{name}_upper")
                stride = fm.F.builtin.scalar_const(index_type, 1, name=f"{name}_step")
                loop = fm.F.control.for_loop(start, stop, stride, value, body=nested, name=f"{name}_loop")
                value = fm.F.tensors.get_item(loop, 0, name=f"{name}_item")
        value = builder.call("math.silu", (value,), tensor, id=f"{name}_tail")
        return builder.region(name, (iv, carried), (value,), iter_count=1)

    inner = region("inner") if inner_upper is not None else None
    body = region("body", inner)
    value = builder.var("input", tensor, id="main_input")
    parameters = [value]
    with construction_scope(builder):
        start = fm.F.builtin.scalar_const(index_type, lower, name="lower")
        if upper is None:
            stop = builder.var("upper", index_type, id="upper")
            parameters.append(stop)
        else:
            stop = fm.F.builtin.scalar_const(index_type, upper, name="upper")
        stride = fm.F.builtin.scalar_const(index_type, step, name="step")
        loop = fm.F.control.for_loop(start, stop, stride, value, body=body, name="main_head")
        output = fm.F.tensors.get_item(loop, 0, name="main_tail")
    builder.function("main", parameters, (output,))
    return fm.verify_module(builder.build(entry="main"))


def _select(monkeypatch, module, before, after, *, grow=None, budget=0):
    baseline = fm.make_buffer_plan(module)
    candidate = replace(baseline)
    if grow is not None:
        function_name, amount = grow
        functions = []
        for function in baseline.functions:
            pools = tuple(
                replace(pool, scope_bytes=pool.scope_bytes + amount)
                if function.name == function_name and pool.memory_space == "workspace"
                else pool
                for pool in function.memory_pools
            )
            functions.append(replace(function, memory_pools=pools))
        candidate = replace(candidate, functions=tuple(functions))

    def sequence(scopes):
        # Isolate schedule selection from hazard discovery using typed plans.
        # The existing emitted-rendezvous test independently checks primitive
        # expansion; these cases check execution multiplicity and control flow.
        return MemorySynchronizationPlan(
            tuple(
                SynchronizationEvent(name, f"{name}_head", f"{name}_tail", scope, (), ())
                for name, values in scopes.items()
                for scope in values
            )
        )

    monkeypatch.setattr(
        synchronization,
        "plan_memory_synchronization",
        lambda _module, plan: sequence(before if plan is baseline else after),
    )
    return dominates_memory_schedule(module, candidate, baseline, bytes_budget=budget)


def test_repeated_callee_savings_can_pay_for_one_extra_entry_rendezvous(monkeypatch):
    module = _call_module({"worker": (), "main": ("worker",) * 3})
    assert _select(monkeypatch, module, {"worker": ("device",)}, {"main": ("device",)})


def test_nested_calls_multiply_and_rejoin_at_the_entry(monkeypatch):
    module = _call_module({"leaf": (), "middle": ("leaf",) * 2, "main": ("middle",) * 3 + ("leaf",)})
    assert _select(monkeypatch, module, {"leaf": ("device",)}, {"main": ("device",) * 6})
    assert not _select(monkeypatch, module, {"leaf": ("device",)}, {"main": ("device",) * 7})
    assert not _select(monkeypatch, module, {"leaf": ("device",)}, {"main": ("device",) * 8})


def test_entry_name_defines_execution_and_unreachable_cost_cannot_pay(monkeypatch):
    module = _call_module({"worker": (), "main": (), "decode": ("worker",) * 3}, entry="decode")
    assert _select(monkeypatch, module, {"worker": ("device",)}, {"decode": ("device",), "main": ("device",) * 100})
    assert not _select(monkeypatch, module, {"main": ("device",) * 100}, {"decode": ("device",)})
    assert not _select(monkeypatch, module, {"main": ("device",)}, {})


@pytest.mark.parametrize(
    "lower,upper,step,accepted",
    [(0, 0, 1, False), (0, 1, 1, False), (0, 3, 1, True), (2, 7, 2, True), (5, 2, 1, False)],
)
def test_static_loop_cost_uses_exact_nonnegative_trip_count(monkeypatch, lower, upper, step, accepted):
    module = _loop_module(lower=lower, upper=upper, step=step)
    assert _select(monkeypatch, module, {"body": ("device",)}, {"main": ("device",) * 2}) is accepted


def test_nested_loop_cost_multiplies_body_execution(monkeypatch):
    module = _loop_module(upper=3, inner_upper=2)
    assert _select(monkeypatch, module, {"inner": ("device",)}, {"main": ("device",) * 5})
    assert not _select(monkeypatch, module, {"inner": ("device",)}, {"main": ("device",) * 6})


@pytest.mark.parametrize(
    "before,after,accepted",
    [
        ({"body": ("device",)}, {"body": ("grid",)}, True),
        ({"body": ("device",)}, {"main": ("device",)}, False),
        ({"main": ("device",)}, {"body": ("device",)}, False),
        ({"main": ("device",)}, {"main": ("grid",)}, True),
    ],
)
def test_dynamic_loop_cost_proves_nonincrease_for_every_trip_count(monkeypatch, before, after, accepted):
    assert _select(monkeypatch, _loop_module(upper=None), before, after) is accepted


def test_dynamic_loop_composes_mandatory_calls_before_bounding_trips(monkeypatch):
    module = _loop_module(upper=None, body_calls=2)
    assert _select(monkeypatch, module, {"worker": ("device",)}, {"body": ("device",)})
    assert not _select(monkeypatch, module, {"worker": ("device",)}, {"body": ("device",), "main": ("device",)})


def test_zero_trip_body_savings_do_not_count_as_an_improvement(monkeypatch):
    assert not _select(monkeypatch, _loop_module(upper=0), {"body": ("device",)}, {})


def test_pool_budget_is_enforced_even_for_unreachable_functions(monkeypatch):
    module = _call_module({"worker": (), "unused": (), "main": ("worker",) * 3})
    assert not _select(monkeypatch, module, {"worker": ("device",)}, {"main": ("device",)}, grow=("unused", 64))
    assert _select(monkeypatch, module, {"worker": ("device",)}, {"main": ("device",)}, grow=("unused", 64), budget=64)
