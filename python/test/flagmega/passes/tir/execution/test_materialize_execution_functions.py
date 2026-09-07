# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace
from copy import deepcopy

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.bufferization import LEGACY_BUFFER_PLAN_SCHEMA
from triton.flagmega.passes.tir import (
    bind_prim_function_buffers,
    materialize_execution_functions,
    materialize_kernel_prim_functions,
)

from .helpers import kernel_attrs, nested_allocated_module, scheduled_nested_module


def test_materializes_one_typed_execution_function_per_graph_function():
    result = scheduled_nested_module()

    assert [value.name for value in result.execution_functions] == [
        "worker",
        "main",
    ]
    worker = result.execution_function_map["worker"]
    main = result.execution_function_map["main"]
    assert [call.call_id for call in fm.execution_calls_of(worker)] == [
        "worker_result"
    ]
    assert [call.call_id for call in fm.execution_calls_of(main)] == [
        "prepared",
        "nested",
        "output",
    ]
    nested = fm.execution_calls_of(main)[1]
    assert nested.callee == "worker"
    assert nested.arguments[0].formal == worker.parameters[0]
    assert nested.results[0].formal == worker.results[0]
    record = fm.verify_buffer_plan(result).call_map[nested.call_id]
    assert nested.workspace_offset == record.workspace_offset
    assert nested.workspace_nbytes == record.workspace_bytes
    assert nested.dependencies == ("prepared",)
    assert fm.execution_calls_of(main)[2].dependencies == ("nested",)


def test_call_bindings_are_concrete_buffer_plan_identities():
    result = scheduled_nested_module()
    plan = fm.verify_buffer_plan(result)

    for function in result.execution_functions:
        for call in fm.execution_calls_of(function):
            for binding in (*call.arguments, *call.results, *call.workspaces):
                assert binding.actual in plan.buffer_map


def test_materialization_is_an_explicit_single_use_pass():
    scheduled = scheduled_nested_module()

    try:
        materialize_execution_functions(scheduled)
    except IRVerificationError as error:
        assert "resume from its output stage" in str(error)
    else:
        raise AssertionError("re-materializing an edited schedule must fail")


def test_verifier_rejects_execution_call_to_missing_target():
    result = scheduled_nested_module()
    main = result.execution_function_map["main"]
    calls = list(main.body.fields)
    calls[0] = replace(calls[0], kernel="missing")
    edited = replace(
        result,
        execution_functions=tuple(
            replace(value, body=fm.T.sequential(tuple(calls)))
            if value.name == "main"
            else value
            for value in result.execution_functions
        ),
    )

    try:
        fm.verify_module(edited)
    except IRVerificationError as error:
        assert "missing kernel definition" in str(error)
    else:
        raise AssertionError("an execution schedule cannot call an unknown target")


def test_codegen_call_abi_does_not_recover_order_from_graph(monkeypatch):
    import triton.flagmega.codegen.triton.call_abi as call_abi

    result = scheduled_nested_module()
    monkeypatch.setattr(
        call_abi,
        "function_nodes",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("graph schedule must not be read")
        ),
    )

    described = call_abi.describe_function_call_abi(result)

    assert [(value["kind"], value["call"]) for value in described["events"]] == [
        ("kernel_call", "prepared"),
        ("function_call", "nested"),
        ("kernel_call", "output"),
    ]
    assert described["events"][1]["memory_pools"] == [
        {
            "memory_space": pool.memory_space,
            "allocation": pool.allocation,
            "offset": pool.offset,
            "scope_bytes": pool.scope_bytes,
            "scope_count": 1,
        }
        for pool in fm.verify_buffer_plan(result).call_map["nested"].memory_pools
    ]


def test_verifier_rejects_agent_schedule_that_moves_consumer_before_producer():
    result = scheduled_nested_module()
    main = result.execution_function_map["main"]
    prepared, nested, output = main.body.fields
    edited_main = replace(
        main,
        body=fm.T.sequential((nested, prepared, output)),
    )
    edited = replace(
        result,
        execution_functions=tuple(
            edited_main if value.name == "main" else value
            for value in result.execution_functions
        ),
    )

    try:
        fm.verify_module(edited)
    except IRVerificationError as error:
        assert "before dependencies" in str(error)
        assert "prepared" in str(error)
    else:
        raise AssertionError("an edited schedule must preserve dataflow order")


def test_materialization_preserves_effect_resource_order_without_data_edge():
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    value_type = fm.tensor_type("bfloat16", (1, 16))
    source = builder.var("source", value_type, id="source")
    writer = builder.call(
        "tir.kernel",
        (source,),
        value_type,
        id="writer",
        effect=fm.effect("write", "state"),
        attrs=kernel_attrs("tir.silu.writer"),
    )
    reader = builder.call(
        "tir.kernel",
        (source,),
        value_type,
        id="reader",
        effect=fm.effect("read", "state"),
        attrs=kernel_attrs("tir.silu.reader"),
    )
    builder.function("main", (source,), (writer, reader))
    selected = materialize_kernel_prim_functions(
        builder.build(entry="main")
    )
    plan = fm.make_buffer_plan(selected)
    bound = bind_prim_function_buffers(selected, plan=plan)
    allocated = replace(
        bound,
        stage="allocated_tir",
        dialect="bufferized_tir",
        metadata={**bound.metadata, "buffer_plan": plan.to_data()},
    )

    calls = fm.execution_calls_of(materialize_execution_functions(allocated).execution_function_map["main"])

    assert calls[0].dependencies == ()
    assert calls[1].dependencies == ("writer",)


def _as_legacy_v5_buffer_plan(data):
    legacy = deepcopy(data)
    legacy["schema"] = LEGACY_BUFFER_PLAN_SCHEMA
    legacy.pop("default_workspace")
    for function in legacy["functions"]:
        pool = next(
            value
            for value in function.pop("memory_pools")
            if value["memory_space"] == "workspace"
        )
        function.update({
            "workspace_bytes": pool["scope_bytes"],
            "workspace_alignment": pool["alignment"],
            "allocations": pool["allocations"],
        })
        for call in function["calls"]:
            pools = call.pop("memory_pools")
            workspace = next(
                (
                    value
                    for value in pools
                    if value["memory_space"] == "workspace"
                ),
                None,
            )
            call.update({
                "workspace_allocation": (
                    None if workspace is None else workspace["allocation"]
                ),
                "workspace_offset": 0 if workspace is None else workspace["offset"],
                "workspace_bytes": (
                    0 if workspace is None else workspace["scope_bytes"]
                ),
            })
    return legacy


def _scheduled_nested_module_with_a_nonempty_callee_pool():
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    value_type = fm.tensor_type("bfloat16", (1, 16))
    worker_input = builder.var("worker_input", value_type, id="worker_input")
    temporary = builder.call(
        "tir.kernel",
        (worker_input,),
        value_type,
        id="worker_temporary",
        attrs=kernel_attrs("tir.silu.worker_temporary"),
    )
    worker_result = builder.call(
        "tir.kernel",
        (temporary,),
        value_type,
        id="worker_result",
        attrs=kernel_attrs("tir.silu.worker_result"),
    )
    builder.function(
        "worker",
        (worker_input,),
        (worker_result,),
        attrs={"calling_convention": "device", "reusable": True},
    )
    source = builder.var("source", value_type, id="source")
    nested = builder.call(
        "tir.call",
        (source,),
        value_type,
        id="nested",
        attrs={"callee": "worker"},
    )
    builder.function("main", (source,), (nested,))
    selected = materialize_kernel_prim_functions(builder.build(entry="main"))
    plan = fm.make_buffer_plan(selected)
    bound = bind_prim_function_buffers(selected, plan=plan)
    allocated = replace(
        bound,
        stage="allocated_tir",
        dialect="bufferized_tir",
        metadata={**bound.metadata, "buffer_plan": plan.to_data()},
    )
    return materialize_execution_functions(allocated)


def test_legacy_v5_execution_workspace_frame_resumes_without_allocation_id():
    result = _scheduled_nested_module_with_a_nonempty_callee_pool()
    main = result.execution_function_map["main"]
    fields = list(main.body.fields)
    nested = fields[0]
    assert nested.call_id == "nested"
    fields[0] = replace(
        nested,
        memory_pools=(
            fm.T.memory_pool_frame(
                "workspace",
                None,
                nested.workspace_offset,
                nested.workspace_nbytes,
            ),
        ),
    )
    resumed = replace(
        result,
        execution_functions=tuple(
            replace(value, body=fm.T.sequential(tuple(fields)))
            if value.name == "main"
            else value
            for value in result.execution_functions
        ),
        metadata={
            **result.metadata,
            "buffer_plan": _as_legacy_v5_buffer_plan(
                fm.verify_buffer_plan(result).to_data()
            ),
        },
    )

    assert fm.verify_module(resumed) is resumed

    wrong = replace(
        fields[0],
        memory_pools=(
            fm.T.memory_pool_frame(
                "workspace",
                None,
                nested.workspace_offset + 256,
                nested.workspace_nbytes,
            ),
        ),
    )
    wrong_main = replace(main, body=fm.T.sequential((wrong,)))
    corrupted = replace(
        resumed,
        execution_functions=tuple(
            wrong_main if value.name == "main" else value
            for value in resumed.execution_functions
        ),
    )
    with pytest.raises(IRVerificationError, match="memory-pool frames differ"):
        fm.verify_module(corrupted)
