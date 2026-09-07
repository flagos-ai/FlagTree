# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Caller liveness must constrain optional callee input/result aliasing."""

import pytest

from triton.flagmega import ir as fm


def _graph(*, retained=False, input_source=False, view=False, nested=False, second_call=False):
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    ty = fm.tensor_type("bfloat16", (1, 128))
    param = builder.var("param", ty, id="param")
    result = builder.call("math.add", (param, param), ty, id="updated")
    builder.function("worker", (param,), (result,))
    callee = "worker"
    if nested:
        outer_param = builder.var("outer_param", ty, id="outer_param")
        outer_result = builder.call("tir.call", (outer_param,), ty, id="outer_result", attrs={"callee": "worker"})
        builder.function("outer", (outer_param,), (outer_result,))
        callee = "outer"
    value = builder.var("value", ty, id="value")
    prepared = value if input_source else builder.call("test.unary", (value,), ty, id="prepared")
    argument = (builder.call("tir.buffer_view", (prepared,), ty, id="view", attrs={"alias_kind": "reshape"}) if view else prepared)
    first = builder.call("tir.call", (argument,), ty, id="first", attrs={"callee": callee})
    outputs = [first]
    if second_call:
        outputs.append(builder.call("tir.call", (argument,), ty, id="second", attrs={"callee": callee}))
    if retained:
        outputs.append(prepared)
    builder.function("main", (value,), outputs)
    return builder.build(entry="main")


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("view", [False, True])
def test_observable_caller_input_prevents_optional_callee_alias(nested, view):
    plan = fm.make_buffer_plan(_graph(retained=True, nested=nested, view=view))
    assert plan.function_map["worker"].result_aliases == ()
    assert not plan.buffer_map["param"].mem_span.may_alias(plan.buffer_map["updated"].mem_span)
    assert not plan.buffer_map["prepared"].mem_span.may_alias(plan.buffer_map["first"].mem_span)


@pytest.mark.parametrize("nested", [False, True])
def test_readonly_entry_argument_is_not_donated_to_nested_calls(nested):
    plan = fm.make_buffer_plan(_graph(input_source=True, nested=nested))
    assert plan.function_map["worker"].result_aliases == ()


def test_one_reusable_callee_contract_accounts_for_every_call_site():
    plan = fm.make_buffer_plan(_graph(second_call=True))
    assert plan.function_map["worker"].result_aliases == ()
    assert len(plan.function_map["main"].calls) == 2


@pytest.mark.parametrize("nested", [False, True])
def test_dead_owned_inputs_still_allow_inplace_and_function_reuse(nested):
    plan = fm.make_buffer_plan(_graph(nested=nested))
    assert plan.function_map["worker"].result_aliases == (("updated", "param"),)
    assert plan.buffer_map["param"].mem_span.must_alias(plan.buffer_map["updated"].mem_span)


def test_identity_call_alias_is_tracked_when_the_original_input_is_consumed():
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    ty = fm.tensor_type("bfloat16", (1, 128))
    param = builder.var("param", ty, id="param")
    updated = builder.call("math.add", (param, param), ty, id="updated")
    builder.function("worker", (param,), (updated,))
    identity_param = builder.var("identity_param", ty, id="identity_param")
    builder.function("identity", (identity_param,), (identity_param,))
    value = builder.var("value", ty, id="value")
    prepared = builder.call("test.unary", (value,), ty, id="prepared")
    preserved = builder.call("tir.call", (prepared,), ty, id="preserved", attrs={"callee": "identity"})
    consumed = builder.call("tir.call", (prepared,), ty, id="consumed", attrs={"callee": "worker"})
    builder.function("main", (value,), (preserved, consumed))
    plan = fm.make_buffer_plan(builder.build(entry="main"))
    assert plan.function_map["identity"].result_aliases == (("identity_param", "identity_param"),)
    assert plan.function_map["worker"].result_aliases == ()


def test_unrelated_tuple_field_liveness_does_not_prevent_input_donation():
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    ty = fm.tensor_type("bfloat16", (1, 128))
    param = builder.var("param", ty, id="param")
    updated = builder.call("math.add", (param, param), ty, id="updated")
    builder.function("worker", (param,), (updated,))
    value = builder.var("value", ty, id="value")
    pair = builder.call("test.pair", (value,), fm.TupleType((ty, ty)), id="pair")
    first = builder.call("builtin.get_item", (pair,), ty, id="first", attrs={"index": 0})
    consumed = builder.call("tir.call", (first,), ty, id="consumed", attrs={"callee": "worker"})
    second = builder.call("builtin.get_item", (pair,), ty, id="second", attrs={"index": 1})
    builder.function("main", (value,), (consumed, second))
    plan = fm.make_buffer_plan(builder.build(entry="main"))
    assert plan.function_map["worker"].result_aliases == (("updated", "param"),)
