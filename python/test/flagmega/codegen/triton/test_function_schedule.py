# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.function_schedule import (
    describe_function_schedule,
)
from triton.flagmega.passes.tir import (
    bind_prim_function_buffers,
    materialize_kernel_prim_functions,
)


def _attrs(
    candidate: str,
    *,
    semantic_op: str = "math.silu",
    facts: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "semantic_op": semantic_op,
        "candidate": candidate,
        "parameters": {"family": "test", "variant": "local"},
        "facts": facts or {},
        "semantic_attrs": {},
    }


def _bufferize(builder: fm.IRBuilder) -> fm.IRModule:
    selected = materialize_kernel_prim_functions(builder.build(entry="main"))
    plan = fm.make_buffer_plan(selected)
    bound = bind_prim_function_buffers(selected)
    return fm.verify_module(replace(
        bound,
        stage="bufferized_tir",
        dialect="bufferized_tir",
        metadata={**bound.metadata, "buffer_plan": plan.to_data()},
    ))


def _mixed_schedule_module() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    value_type = fm.tensor_type("bfloat16", (1, 16))
    source = builder.var("source", value_type, id="source")
    local_a = builder.call(
        "tir.kernel",
        (source,),
        value_type,
        id="local_a",
        attrs=_attrs("tir.local_a"),
    )
    synchronized = builder.call(
        "tir.kernel",
        (local_a,),
        value_type,
        id="synchronized",
        attrs=_attrs(
            "tir.synchronized",
            facts={"requires": ("grid_sync",)},
        ),
    )
    local_b = builder.call(
        "tir.kernel",
        (synchronized,),
        value_type,
        id="local_b",
        attrs=_attrs("tir.local_b"),
    )
    collective = builder.call(
        "tir.kernel",
        (local_b,),
        value_type,
        id="collective",
        attrs=_attrs(
            "tir.collective",
            semantic_op="distributed.boxing",
            facts={"collective_semantics": "gather_reduce_scatter"},
        ),
    )
    local_c = builder.call(
        "tir.kernel",
        (collective,),
        value_type,
        id="local_c",
        attrs=_attrs("tir.local_c"),
    )
    builder.function(
        "main",
        (source,),
        (local_c,),
        attrs={"noinline": True, "reusable": True},
    )
    return _bufferize(builder)


def test_schedule_splits_only_on_explicit_synchronization_semantics():
    schedule = describe_function_schedule(_mixed_schedule_module())

    assert schedule["physical_strategy"] == "direct_noinline"
    assert schedule["direct_noinline_safe"] is True
    assert [region["kind"] for region in schedule["regions"]] == [
        "local_segment",
        "synchronized_local_kernel",
        "local_segment",
        "collective_kernel",
        "local_segment",
    ]
    assert schedule["regions"][0]["calls"] == ("local_a",)
    assert schedule["regions"][0]["owner_participation"] == "all"
    assert schedule["regions"][1]["barrier_owner"] == "inline_kernel_body"
    assert schedule["regions"][1]["barrier_before"] is False
    assert schedule["regions"][3]["barrier_owner"] == "function_schedule"
    assert schedule["regions"][3]["barrier_before"] is True


def test_ordinary_local_segment_is_safe_to_reuse_as_one_device_function():
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    value_type = fm.tensor_type("bfloat16", (1, 16))
    source = builder.var("source", value_type, id="source")
    first = builder.call(
        "tir.kernel",
        (source,),
        value_type,
        id="first",
        attrs=_attrs("tir.first"),
    )
    second = builder.call(
        "tir.kernel",
        (first,),
        value_type,
        id="second",
        attrs=_attrs("tir.second"),
    )
    builder.function(
        "main",
        (source,),
        (second,),
        attrs={"noinline": True, "reusable": True},
    )

    schedule = describe_function_schedule(_bufferize(builder))

    assert schedule["physical_strategy"] == "direct_noinline"
    assert schedule["direct_noinline_safe"] is True
    assert schedule["regions"] == [{
        "kind": "local_segment",
        "name": "main_local_0",
        "calls": ("first", "second"),
        "execution_domain": "dense_local_shard",
        "owner_participation": "all",
        "lowering": "function_body",
        "barrier_before": False,
        "memory_dependencies": (),
    }]


def test_memory_hazard_splits_regions_without_expanding_function_calls():
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    value_type = fm.tensor_type("bfloat16", (1, 16))
    source = builder.var("source", value_type, id="source")
    first = builder.call(
        "tir.kernel",
        (source,),
        value_type,
        id="first",
        attrs=_attrs("tir.first"),
    )
    second = builder.call(
        "tir.kernel",
        (first,),
        value_type,
        id="second",
        attrs=_attrs("tir.second"),
    )
    builder.function(
        "main",
        (source,),
        (second,),
        attrs={"noinline": True, "reusable": True},
    )
    module = _bufferize(builder)
    dependency = {
        "function": "main",
        "after": "first",
        "before": "second",
        "scope": "grid",
        "hazards": ("WRITE->READ",),
        "ranges": (),
    }
    module = replace(module, metadata={
        **module.metadata,
        "memory_synchronization": {
            "schema": "flagmega.memory-synchronization/v1",
            "events": (dependency,),
        },
    })

    schedule = describe_function_schedule(module)

    assert schedule["requires_synchronization"] is True
    assert schedule["direct_noinline_safe"] is True
    assert schedule["physical_strategy"] == "direct_noinline"
    assert [region["calls"] for region in schedule["regions"]] == [
        ("first",),
        ("second",),
    ]
    assert [region["barrier_before"] for region in schedule["regions"]] == [
        False,
        True,
    ]
    assert schedule["regions"][1]["memory_dependencies"] == (dependency,)
