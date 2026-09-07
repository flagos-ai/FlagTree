# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.tir import materialize_kernel_prim_functions
from triton.flagmega.passes.tir.bufferize import plan_memory_synchronization


def _two_cache_updates(*, second_layer: int) -> fm.IRModule:
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    slots_type = fm.tensor_type("bfloat16", (1, 2, 8))
    scalar_i32 = fm.tensor_type("int32", ())
    scalar_bool = fm.tensor_type("bool", ())
    state_type = fm.RefType("state", (("cache", slots_type),))
    slots = builder.var("slots", slots_type, id="slots")
    state = builder.var("state", state_type, id="state")
    layer0 = builder.call(
        "builtin.scalar_const", (), scalar_i32, id="layer0", attrs={"value": 0}
    )
    layer1 = builder.call(
        "builtin.scalar_const", (), scalar_i32, id="layer1",
        attrs={"value": second_layer},
    )
    advance = builder.call(
        "builtin.scalar_const", (), scalar_bool, id="advance",
        attrs={"value": False},
    )
    attrs = {
        "semantic_op": "nn.update_paged_attention_kv_cache",
        "candidate": "tir.update_cache.unit",
        "parameters": {"family": "update_cache", "variant": "unit"},
        "facts": {},
        "semantic_attrs": {"cache_kind": "key", "layout": ("seq", "head", "dim")},
    }
    first = builder.call(
        "tir.kernel", (slots, state, layer0, advance), state_type,
        id="first", effect=fm.effect("read_write", "cache"), attrs=attrs,
    )
    second = builder.call(
        "tir.kernel", (slots, state, layer1, advance), state_type,
        id="second", effect=fm.effect("read_write", "cache"), attrs=attrs,
    )
    builder.function("main", (slots, state), (first, second))
    return materialize_kernel_prim_functions(builder.build(entry="main"))


def test_distinct_static_resource_partitions_do_not_alias():
    module = _two_cache_updates(second_layer=1)

    plan = plan_memory_synchronization(module, fm.make_buffer_plan(module))

    assert plan.events == ()


def test_equal_static_resource_partitions_require_chip_synchronization():
    module = _two_cache_updates(second_layer=0)

    plan = plan_memory_synchronization(module, fm.make_buffer_plan(module))

    assert len(plan.events) == 1
    assert plan.events[0].scope == "grid"
    assert plan.events[0].axis_group_axes == ()
