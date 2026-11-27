# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega.ir import (
    DType,
    IRBuilder,
    TupleType,
    make_buffer_plan,
    tensor_type,
    verify_buffer_plan,
)


def _linear_module():
    builder = IRBuilder(dialect="tir", stage="selected_tir")
    value_type = tensor_type(DType.BFLOAT16, [1, 128])
    source = builder.var("source", value_type, id="source")
    first = builder.call("test.unary", [source], value_type, id="first")
    second = builder.call("test.unary", [first], value_type, id="second")
    third = builder.call("test.unary", [second], value_type, id="third")
    output = builder.call("test.unary", [third], value_type, id="output")
    builder.function("main", [source], [output])
    return builder.build(entry="main")


def test_workspace_reuses_storage_after_the_last_use():
    module = _linear_module()
    plan = make_buffer_plan(module)
    buffers = {buffer.id: buffer for buffer in plan.buffers}

    assert buffers["first"].offset == buffers["third"].offset
    assert buffers["first"].live_end < buffers["third"].live_start
    assert plan.workspace_bytes == 512

    bufferized = replace(
        module,
        stage="bufferized_tir",
        metadata={**dict(module.metadata), "buffer_plan": plan.to_data()},
    )
    assert verify_buffer_plan(bufferized) == plan


def test_workspace_keeps_an_input_distinct_from_its_consuming_output():
    plan = make_buffer_plan(_linear_module())
    buffers = {buffer.id: buffer for buffer in plan.buffers}

    assert buffers["first"].live_end == buffers["second"].live_start
    assert buffers["first"].offset != buffers["second"].offset
    assert buffers["second"].offset != buffers["third"].offset


def test_workspace_applies_a_named_inplace_parameter_after_its_last_use():
    builder = IRBuilder(dialect="tir", stage="selected_tir")
    value_type = tensor_type(DType.BFLOAT16, [1, 128])
    source = builder.var("source", value_type, id="source")
    lhs = builder.call("test.unary", [source], value_type, id="lhs")
    rhs = builder.call("test.unary", [source], value_type, id="rhs")
    combined = builder.call("math.add", [lhs, rhs], value_type, id="combined")
    output = builder.call("test.unary", [combined], value_type, id="output")
    builder.function("main", [source], [output])

    module = builder.build(entry="main")
    plan = make_buffer_plan(module)
    buffers = {buffer.id: buffer for buffer in plan.buffers}

    assert buffers["combined"].alias_of == "lhs"
    assert buffers["combined"].offset == buffers["lhs"].offset
    assert buffers["combined"].live_start == buffers["lhs"].live_end
    bufferized = replace(
        module,
        stage="bufferized_tir",
        metadata={**dict(module.metadata), "buffer_plan": plan.to_data()},
    )
    assert verify_buffer_plan(bufferized) == plan


def test_workspace_rejects_inplace_reuse_while_the_parameter_has_a_future_use():
    builder = IRBuilder(dialect="tir", stage="selected_tir")
    value_type = tensor_type(DType.BFLOAT16, [1, 128])
    source = builder.var("source", value_type, id="source")
    lhs = builder.call("test.unary", [source], value_type, id="lhs")
    rhs = builder.call("test.unary", [source], value_type, id="rhs")
    combined = builder.call("math.add", [lhs, rhs], value_type, id="combined")
    future_lhs = builder.call("test.unary", [lhs], value_type, id="future_lhs")
    output = builder.call(
        "test.binary", [combined, future_lhs], value_type, id="output"
    )
    builder.function("main", [source], [output])

    plan = make_buffer_plan(builder.build(entry="main"))
    buffers = {buffer.id: buffer for buffer in plan.buffers}

    assert buffers["combined"].alias_of is None
    assert buffers["combined"].offset != buffers["lhs"].offset


def test_tuple_field_lifetime_follows_get_item_consumers():
    builder = IRBuilder(dialect="tir", stage="selected_tir")
    value_type = tensor_type(DType.BFLOAT16, [1, 128])
    source = builder.var("source", value_type, id="source")
    pair = builder.call(
        "test.pair",
        [source],
        TupleType((value_type, value_type)),
        id="pair",
    )
    first = builder.call(
        "builtin.get_item", [pair], value_type, id="first", attrs={"index": 0}
    )
    intermediate = builder.call("test.unary", [source], value_type, id="intermediate")
    delayed = builder.call(
        "builtin.get_item", [pair], value_type, id="delayed", attrs={"index": 1}
    )
    output = builder.call("test.binary", [first, delayed, intermediate], value_type, id="output")
    builder.function("main", [source], [output])

    plan = make_buffer_plan(builder.build(entry="main"))
    buffers = {buffer.id: buffer for buffer in plan.buffers}

    assert buffers["pair.0"].live_end == 5
    assert buffers["pair.1"].live_end == 5
    assert buffers["intermediate"].offset not in {
        buffers["pair.0"].offset,
        buffers["pair.1"].offset,
    }


def test_workspace_extends_a_terminal_free_span_instead_of_leaving_a_hole():
    builder = IRBuilder(dialect="tir", stage="selected_tir")
    small_type = tensor_type(DType.BFLOAT16, [1, 128])
    large_type = tensor_type(DType.BFLOAT16, [1, 256])
    source = builder.var("source", small_type, id="source")
    pair = builder.call(
        "test.pair",
        [source],
        TupleType((small_type, small_type)),
        id="pair",
    )
    dead_field = builder.call(
        "builtin.get_item", [pair], small_type, id="dead_field", attrs={"index": 1}
    )
    large = builder.call("test.expand", [source], large_type, id="large")
    live_field = builder.call(
        "builtin.get_item", [pair], small_type, id="live_field", attrs={"index": 0}
    )
    output = builder.call(
        "test.join", [live_field, large], small_type, id="output"
    )
    builder.function("main", [source], [output])

    plan = make_buffer_plan(builder.build(entry="main"))
    buffers = {buffer.id: buffer for buffer in plan.buffers}

    assert buffers["pair.0"].offset == 0
    assert buffers["pair.1"].offset == 256
    assert buffers["large"].offset == 256
    assert plan.workspace_bytes == 768
