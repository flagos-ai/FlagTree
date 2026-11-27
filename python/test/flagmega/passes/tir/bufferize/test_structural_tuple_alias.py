# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm


def test_tuple_and_get_item_forward_existing_physical_buffer_identities():
    tensor = fm.tensor_type("bfloat16", (1, 8))
    builder = fm.IRBuilder(dialect="semantic_tir", stage="packaged_tir")
    lhs = builder.var("lhs", tensor, id="lhs")
    rhs = builder.var("rhs", tensor, id="rhs")
    first = builder.call("math.add", (lhs, rhs), tensor, id="first")
    second = builder.call("math.multiply", (lhs, rhs), tensor, id="second")
    pair = builder.call(
        "builtin.tuple",
        (first, second),
        fm.TupleType((tensor, tensor)),
        id="pair",
    )
    selected = builder.call(
        "builtin.get_item", (pair,), tensor, id="selected", attrs={"index": 1}
    )
    builder.function("main", (lhs, rhs), (selected,))

    plan = fm.make_buffer_plan(builder.build(entry="main"))
    values = dict(plan.function_map["main"].values)

    assert values["pair"] == (values["first"][0], values["second"][0])
    assert values["selected"] == values["second"]
    assert plan.buffer_map[values["selected"][0]].storage == "output"
    assert not any(buffer.source_node == "pair" for buffer in plan.buffers)


def test_tuple_of_vector_reinterpret_views_preserves_each_source_memspan():
    packed = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 1))
    unpacked = fm.tensor_type("bfloat16", (1, 8))
    builder = fm.IRBuilder(dialect="semantic_tir", stage="packaged_tir")
    lhs = builder.var("lhs", packed, id="lhs")
    rhs = builder.var("rhs", packed, id="rhs")
    lhs_view = builder.call(
        "tir.buffer_view",
        (lhs,),
        unpacked,
        id="lhs_view",
        attrs={"alias_kind": "vector_reinterpret"},
    )
    rhs_view = builder.call(
        "tir.buffer_view",
        (rhs,),
        unpacked,
        id="rhs_view",
        attrs={"alias_kind": "vector_reinterpret"},
    )
    pair = builder.call(
        "builtin.tuple",
        (lhs_view, rhs_view),
        fm.TupleType((unpacked, unpacked)),
        id="pair",
    )
    selected = builder.call(
        "builtin.get_item", (pair,), unpacked, id="selected", attrs={"index": 0}
    )
    builder.function("main", (lhs, rhs), (selected,))

    plan = fm.make_buffer_plan(builder.build(entry="main"))
    values = dict(plan.function_map["main"].values)
    lhs_buffer = plan.buffer_map[values["lhs"][0]]
    selected_buffer = plan.buffer_map[values["selected"][0]]

    assert selected_buffer.mem_span.must_alias(lhs_buffer.mem_span)
    assert values["pair"] == (values["lhs_view"][0], values["rhs_view"][0])
