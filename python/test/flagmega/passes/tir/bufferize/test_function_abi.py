# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.ir import DType, IRBuilder, make_buffer_plan, tensor_type


def _two_call_module():
    builder = IRBuilder(dialect="tir", stage="selected_tir")
    value_type = tensor_type(DType.BFLOAT16, [1, 128])

    parameter = builder.var("parameter", value_type, id="callee_parameter")
    temporary = builder.call("test.unary", [parameter], value_type, id="callee_temporary")
    result = builder.call("test.unary", [temporary], value_type, id="callee_result")
    builder.function("worker", [parameter], [result], attrs={"calling_convention": "device"})

    source = builder.var("source", value_type, id="source")
    prepared = builder.call("test.unary", [source], value_type, id="prepared")
    first = builder.call(
        "tir.call", [prepared], value_type, id="first_call", attrs={"callee": "worker"}
    )
    second = builder.call(
        "tir.call", [first], value_type, id="second_call", attrs={"callee": "worker"}
    )
    builder.function("main", [source], [second])
    return builder.build(entry="main")


def test_function_buffer_abi_maps_formals_results_and_sat_call_frames():
    plan = make_buffer_plan(_two_call_module())
    worker = plan.function_map["worker"]
    main = plan.function_map["main"]

    assert worker.workspace_bytes == 256
    assert len(main.calls) == 2
    assert main.calls[0].workspace_bytes == worker.workspace_bytes
    assert main.calls[1].workspace_bytes == worker.workspace_bytes
    assert dict(main.calls[0].arguments)["callee_parameter"] == "prepared"
    assert dict(main.calls[0].results)["callee_result"] == "first_call"

    local = plan.buffer_map["callee_temporary"]
    resolved = plan.resolve_call_buffer("first_call", local.id)
    assert resolved.offset == main.calls[0].workspace_offset + local.offset
    assert resolved.role == "call_memory_pool"
    assert plan.resolve_call_buffer("first_call", "callee_parameter").id == "prepared"
    assert plan.resolve_call_buffer("first_call", "callee_parameter").offset == plan.buffer_map["prepared"].offset
    assert plan.resolve_call_buffer("first_call", "callee_result").offset == plan.buffer_map["first_call"].offset


def test_optional_callee_alias_preserves_a_nonwritable_entry_input():
    builder = IRBuilder(dialect="tir", stage="selected_tir")
    value_type = tensor_type(DType.BFLOAT16, [1, 128])
    parameter = builder.var("parameter", value_type, id="callee_parameter")
    result = builder.call(
        "math.add", [parameter, parameter], value_type, id="callee_result"
    )
    builder.function("worker", [parameter], [result])
    source = builder.var("source", value_type, id="source")
    call = builder.call(
        "tir.call", [source], value_type, id="call", attrs={"callee": "worker"}
    )
    builder.function("main", [source], [call])

    plan = make_buffer_plan(builder.build(entry="main"))
    assert plan.function_map["worker"].result_aliases == ()
    assert not plan.buffer_map["callee_parameter"].mem_span.may_alias(
        plan.buffer_map["callee_result"].mem_span
    )
