# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Instantiate a callee's result alias groups in each caller frame."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError


def result_alias_graph(*, view_first=False, nested=True, dialect="tir", stage="selected_tir"):
    placement = fm.Placement((2, 2), "yx", "bb")
    tensor = fm.tensor_type("float32", (2, 8))
    broadcast = fm.DistributedType(tensor, (fm.SBP.broadcast(),) * 2, placement)
    split = fm.DistributedType(
        tensor, (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,))), placement,
    )

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect=dialect, stage=stage, entry="main")

        def forward(self):
            parameter = self.input("parameter", broadcast, id="parameter")
            produced = fm.F.math.add(parameter, parameter, name="produced")
            other = fm.F.math.mul(parameter, parameter, name="other")
            viewed = fm.F.distributed.sharded_view(produced, split, name="viewed")
            first = fm.F.builtin.tuple(other, produced, name="pair") if nested else produced
            results = (viewed, first) if view_first else (first, viewed)
            result_type = fm.TupleType(tuple(v.type for v in results))
            values = [self.input(f"value{i}", broadcast, id=f"value{i}") for i in range(2)]
            calls = [fm.F.builtin.call(value, result_type=result_type, callee="worker", name=f"call{i}")
                     for i, value in enumerate(values)]
            views = [fm.F.tensors.get_item(c, 0 if view_first else 1, name=f"view{i}")
                     for i, c in enumerate(calls)]
            output = fm.F.math.add(*views, name="output")
            self.function("main", tuple(values), (output,))
            self.function("worker", (parameter,), results, attrs={"reusable": True, "noinline": True})

    module = Graph().build()
    module = replace(module, metadata={"launch_contract": {"grid_mesh": placement.to_data()},
                                       "auto_distribution": {"placement": placement.to_data()}})
    return module


@pytest.mark.parametrize("view_first", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_result_views_share_the_produced_tuple_leaf_in_the_caller(view_first, nested):
    module = result_alias_graph(view_first=view_first, nested=nested)
    plan = fm.make_buffer_plan(module)
    assert plan.buffer_map["produced"].mem_span.must_alias(plan.buffer_map["viewed"].mem_span)
    for name in ("call0", "call1"):
        actuals = dict(plan.call_map[name].results)
        produced = plan.buffer_map[actuals["produced"]]
        viewed = plan.buffer_map[actuals["viewed"]]
        assert produced.mem_span.must_alias(viewed.mem_span), (
            "callee aliases must name the same allocation in its caller", produced, viewed,
        )
        assert produced.distributed_type == module.node_map["produced"].type
        assert viewed.distributed_type == module.node_map["viewed"].type
        if nested:
            assert not produced.mem_span.may_alias(plan.buffer_map[actuals["other"]].mem_span)
    first = plan.buffer_map[dict(plan.call_map["call0"].results)["produced"]]
    second = plan.buffer_map[dict(plan.call_map["call1"].results)["produced"]]
    assert not first.mem_span.may_alias(second.mem_span)


def test_verifier_rejects_a_call_that_splits_one_callee_result_span():
    module = result_alias_graph()
    plan = fm.make_buffer_plan(module)
    call = plan.call_map["call0"]
    unrelated = dict(call.results)["other"]
    forged_call = replace(call, results=tuple(
        (formal, unrelated if formal == "viewed" else actual) for formal, actual in call.results
    ))
    forged = replace(plan, functions=tuple(
        replace(function, calls=tuple(forged_call if c.call == call.call else c for c in function.calls))
        if function.name == module.entry else function for function in plan.functions
    ))
    invalid = replace(module, metadata={**module.metadata, "buffer_plan": forged.to_data()})
    with pytest.raises(IRVerificationError, match="callee.*alias|aliased.*callee"):
        fm.verify_buffer_plan(invalid)


@pytest.mark.parametrize("view_first", [False, True])
@pytest.mark.parametrize("kind", ["reshape", "vector_reinterpret"])
def test_result_aliases_preserve_each_view_type_and_external_ownership(kind, view_first):
    source_type = fm.tensor_type("float32", (2, 8))
    target_type = (fm.tensor_type("float32", (16,)) if kind == "reshape" else
                   fm.tensor_type(fm.vector_type("float32", (4,)), (2, 2)))

    class Graph(fm.Module):
        def forward(self):
            parameter = self.input("parameter", source_type, id="parameter")
            produced = fm.F.math.add(parameter, parameter, name="produced")
            viewed = fm.F.tir.buffer_view(produced, target_type, alias_kind=kind, name="viewed")
            results = (viewed, produced) if view_first else (produced, viewed)
            value = self.input("value", source_type, id="value")
            call = fm.F.builtin.call(value, callee="worker", result_type=fm.TupleType(tuple(v.type for v in results)),
                                     name="call")
            self.function("main", (value,), (call,))
            self.function("worker", (parameter,), results, attrs={"reusable": True, "noinline": True})

    module = Graph(dialect="tir", stage="selected_tir", entry="main").build()
    plan = fm.make_buffer_plan(module)
    actuals = dict(plan.call_map["call"].results)
    produced = plan.buffer_map[actuals["produced"]]
    viewed = plan.buffer_map[actuals["viewed"]]
    assert produced.mem_span.must_alias(viewed.mem_span)
    assert produced.dtype == source_type.dtype and produced.shape == (2, 8)
    assert viewed.dtype == target_type.dtype and viewed.shape == tuple(d.fixed_value for d in target_type.shape)
    assert produced.storage == viewed.storage == "output"
    fm.verify_buffer_plan(replace(module, metadata={"buffer_plan": plan.to_data()}))
