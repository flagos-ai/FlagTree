# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.tir import (
    bind_prim_function_buffers,
    materialize_kernel_prim_functions,
)


def _tuple_kernel() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    tensor = fm.tensor_type("bfloat16", (1, 16))
    state = fm.RefType("state", (("cache", tensor),))
    scalar_i32 = fm.tensor_type("int32", ())
    scalar_bool = fm.tensor_type("bool", ())
    types = (
        tensor,
        state,
        scalar_i32,
        scalar_bool,
    )
    arguments = tuple(
        builder.var(f"arg{index}", value, id=f"arg{index}")
        for index, value in enumerate(types)
    )
    output = builder.call(
        "tir.kernel",
        arguments,
        fm.TupleType((tensor, state)),
        id="attention",
        effect=fm.effect("read_write", "cache"),
        attrs={
            "semantic_op": "nn.update_paged_attention_kv_cache",
            "candidate": "tir.attention.unit",
            "parameters": {"family": "attention", "variant": "unit"},
            "facts": {},
            "semantic_attrs": {},
        },
    )
    builder.function("main", arguments, (output,))
    return builder.build(entry="main")


def test_tuple_kernel_has_one_named_output_parameter_per_field():
    module = materialize_kernel_prim_functions(_tuple_kernel())
    function = module.kernel_definitions[0]
    dispatch = fm.kernel_dispatch_for_call(module, module.node_map["attention"])

    assert tuple(value.name for value in function.output_parameters) == (
        "result_0",
        "result_1",
    )
    assert tuple(value.type for value in function.output_parameters) == (
        module.node_map["attention"].type.fields
    )
    assert tuple(value.storage for value in function.results.values) == (
        "result_0",
        "result_1",
    )
    assert dispatch.outputs == ("result_0", "result_1")
    assert "result_0" in dispatch.writes
    assert "result_1" not in dispatch.writes
    assert "state" in dispatch.writes
    assert dispatch.memory_effect_map["result_1"] is fm.MemoryEffect.NONE
    assert function.runtime_return_type == module.node_map["attention"].type


def test_each_multi_output_parameter_binds_only_its_own_buffer_leaves():
    module = materialize_kernel_prim_functions(_tuple_kernel())
    plan = fm.make_buffer_plan(module)
    bound = bind_prim_function_buffers(module, plan=plan)
    outputs = bound.kernel_definitions[0].output_parameters

    assert len(outputs[0].buffers) == 1
    assert outputs[0].buffers[0].name == "result_0"
    assert tuple(value.name for value in outputs[1].buffers) == (
        "result_1.cache",
    )
