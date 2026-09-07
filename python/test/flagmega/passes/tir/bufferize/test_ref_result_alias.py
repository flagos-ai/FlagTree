# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm


def _chained_ref_update_module() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    cache_type = fm.tensor_type("bfloat16", (2, 8))
    length_type = fm.tensor_type("int32", (1,))
    state_type = fm.RefType(
        "state",
        (("cache", cache_type), ("length", length_type)),
    )

    worker_state = builder.var("state", state_type, id="worker_state")
    worker_updated = builder.call(
        "test.update_state",
        (worker_state,),
        state_type,
        id="worker_updated",
        effect=fm.effect("read_write", "state"),
    )
    builder.function("worker", (worker_state,), (worker_updated,))

    state = builder.var("state", state_type, id="state")
    first = builder.call(
        "tir.call", (state,), state_type,
        id="first", attrs={"callee": "worker"},
    )
    second = builder.call(
        "tir.call", (first,), state_type,
        id="second", attrs={"callee": "worker"},
    )
    builder.function("main", (state,), (second,))
    return builder.build(entry="main")


def test_direct_and_nested_ref_results_preserve_parameter_memspans():
    plan = fm.make_buffer_plan(_chained_ref_update_module())
    worker = plan.function_map["worker"]
    main = plan.function_map["main"]

    worker_parameters = worker.parameters[0][1]
    assert worker.outputs[0][1] == worker_parameters
    assert set(worker.result_aliases) == {
        (buffer, buffer) for buffer in worker_parameters
    }

    entry_state = main.parameters[0][1]
    assert dict(main.values)["first"] == entry_state
    assert dict(main.values)["second"] == entry_state
    assert main.outputs[0][1] == entry_state
    assert all(
        dict(call.results) == dict(call.arguments)
        for call in main.calls
    )
    assert not any(
        descriptor.source_node in {"worker_updated", "first", "second"}
        and descriptor.storage == "workspace"
        for descriptor in plan.buffers
    )
