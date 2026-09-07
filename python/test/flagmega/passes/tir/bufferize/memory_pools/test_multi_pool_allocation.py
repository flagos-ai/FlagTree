# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.call_abi import describe_function_call_abi
from triton.flagmega.codegen.triton.function_call import emit_function_call_arguments
from triton.flagmega.codegen.triton.runtime_binding import (
    describe_function_runtime_binding,
)
from triton.flagmega.passes.tir import (
    bind_prim_function_buffers,
    materialize_execution_functions,
    materialize_kernel_prim_functions,
)
from triton.flagmega.passes.tir.bufferize import BufferizationOptions


def _options() -> BufferizationOptions:
    generic = BufferizationOptions.generic(alignment=64)
    workspace, rdata, external = generic.memory_spaces
    return BufferizationOptions((
        replace(workspace, sharing_scope=fm.MemorySharingScope.CHIP),
        replace(workspace, name="block_data", sharing_scope=fm.MemorySharingScope.BLOCK),
        rdata,
        external,
    ))


def _allocated_nested_module():
    value_type = fm.tensor_type("float32", (16,))
    builder = fm.IRBuilder(
        dialect="tir",
        stage="selected_tir",
        metadata={
            "launch_contract": {
                "grid_mesh": {"hierarchy": [2, 4], "hierarchy_levels": "bb"}
            }
        },
    )
    worker_input = builder.var("worker_input", value_type, id="worker_input")
    worker_temporary = builder.call(
        "tir.kernel",
        (worker_input,),
        value_type,
        id="worker_temporary",
        attrs={
            "semantic_op": "math.silu",
            "candidate": "tir.silu.block_local",
            "parameters": {"family": "elementwise", "variant": "silu"},
            "facts": {},
            "semantic_attrs": {},
        },
        metadata={"bufferization.memory_space": "block_data"},
    )
    worker_result = builder.call(
        "tir.kernel",
        (worker_temporary,),
        value_type,
        id="worker_result",
        attrs={
            "semantic_op": "math.silu",
            "candidate": "tir.silu.result",
            "parameters": {"family": "elementwise", "variant": "silu"},
            "facts": {},
            "semantic_attrs": {},
        },
    )
    builder.function("worker", (worker_input,), (worker_result,))

    source = builder.var("source", value_type, id="source")
    nested = builder.call(
        "tir.call",
        (source,),
        value_type,
        id="nested",
        attrs={"callee": "worker"},
    )
    builder.function("main", (source,), (nested,))
    module = materialize_kernel_prim_functions(builder.build(entry="main"))
    plan = fm.make_buffer_plan(module, options=_options())
    module = bind_prim_function_buffers(module, plan=plan)
    return replace(
        module,
        stage="allocated_tir",
        dialect="bufferized_tir",
        metadata={**module.metadata, "buffer_plan": plan.to_data()},
    )


def test_sat_planner_allocates_target_declared_function_pools_independently():
    module = _allocated_nested_module()
    plan = fm.verify_buffer_plan(module)
    worker = plan.function_map["worker"]
    main = plan.function_map["main"]

    assert worker.memory_pool_map["workspace"].scope_bytes == 0
    assert worker.memory_pool_map["block_data"].scope_bytes == 64
    assert worker.memory_pool_map["block_data"].allocations == (
        plan.buffer_map["worker_temporary"].physical_id,
    )
    assert main.calls[0].memory_pool_map["block_data"].scope_bytes == 64
    assert main.calls[0].memory_pool_map["block_data"].allocation in (
        main.memory_pool_map["block_data"].allocations
    )
    assert plan.workspace_bytes == 0


def test_multi_pool_call_frames_survive_execution_ir_and_call_abi():
    scheduled = materialize_execution_functions(_allocated_nested_module())
    call = fm.execution_calls_of(scheduled.execution_function_map["main"])[0]
    event = describe_function_call_abi(scheduled)["events"][0]

    assert tuple(frame.memory_space for frame in call.memory_pools) == (
        "block_data",
    )
    assert event["memory_pools"] == [{
        "memory_space": "block_data",
        "allocation": call.memory_pools[0].allocation,
        "offset": call.memory_pools[0].offset,
        "scope_bytes": 64,
        "scope_count": 8,
    }]


def test_nested_block_pool_is_passed_as_one_scope_local_frame():
    scheduled = materialize_execution_functions(_allocated_nested_module())
    caller = describe_function_runtime_binding(scheduled, function_name="main")
    callee = describe_function_runtime_binding(scheduled, function_name="worker")
    event = caller["call_abi"]["events"][0]
    arguments = emit_function_call_arguments(scheduled, caller, event)
    caller_pool = next(
        pool for pool in caller["pools"] if pool["storage"] == "block_data"
    )
    callee_pool = next(
        pool for pool in callee["pools"] if pool["storage"] == "block_data"
    )

    assert caller_pool["scope_count"] == 8
    assert caller_pool["nbytes"] == 8 * 64
    assert callee_pool["scope_count"] == 1
    assert callee_pool["scope_local"] is True
    assert callee_pool["nbytes"] == 64
    assert arguments[-1] == "_flagmega_block_scope_base(block_data, 64)"
    worker_kernel_abi = callee["call_abi"]["kernel_calls"][0]["outputs"][0][
        "buffers"
    ][0]["abi"]
    assert worker_kernel_abi["pool_scope_count"] == 1
    assert worker_kernel_abi["pool_scope_local"] is True
    assert "pool_scope_stride_bytes" not in worker_kernel_abi
