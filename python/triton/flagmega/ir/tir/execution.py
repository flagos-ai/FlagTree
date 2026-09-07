# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Traversal and verification helpers for execution functions."""

from __future__ import annotations

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.tir.execution_function import ExecutionFunction
from triton.flagmega.ir.tir.barrier import Barrier
from triton.flagmega.ir.tir.pipeline_stage import PipelineStage
from triton.flagmega.ir.tir.prim_function_call import PrimFunctionCall
from triton.flagmega.ir.tir.kernel_invoke import KernelInvoke
from triton.flagmega.ir.tir.producer_consumer_region import ProducerConsumerRegion
from triton.flagmega.ir.tir.visitor import iter_tir_children
from triton.flagmega.ir.tir.kernel_dispatch import kernel_dispatch_of
from triton.flagmega.ir.tir.shared_call_abi import verify_shared_call_abi


def execution_calls_of(function: ExecutionFunction) -> tuple[PrimFunctionCall | KernelInvoke, ...]:
    """Return semantic calls once, following the consumer role of a region."""

    result: list[PrimFunctionCall | KernelInvoke] = []

    def visit(node) -> None:
        if isinstance(node, (PrimFunctionCall, KernelInvoke)):
            result.append(node)
            return
        if isinstance(node, PipelineStage):
            if isinstance(node.operation, (PrimFunctionCall, KernelInvoke)):
                result.append(node.operation)
            return
        if isinstance(node, ProducerConsumerRegion):
            visit(node.consume_body)
            return
        for child in iter_tir_children(node):
            visit(child)

    visit(function.body)
    return tuple(result)


def execution_items_of(function: ExecutionFunction):
    """Return calls and barriers once in semantic consumer order."""

    result = []

    def visit(node) -> None:
        if isinstance(node, (PrimFunctionCall, KernelInvoke, Barrier)):
            result.append(node)
            return
        if isinstance(node, PipelineStage):
            if isinstance(node.operation, (PrimFunctionCall, KernelInvoke)):
                result.append(node.operation)
            return
        if isinstance(node, ProducerConsumerRegion):
            visit(node.consume_body)
            return
        for child in iter_tir_children(node):
            visit(child)

    visit(function.body)
    return tuple(result)


def verify_execution_functions(module) -> None:
    functions = module.execution_function_map
    if len(functions) != len(module.execution_functions):
        raise IRVerificationError(
            "ExecutionFunction names must be unique.", stage=module.stage
        )
    if functions:
        execution_names = set(functions)
        graph_names = set(module.function_map)
        fragment = module.metadata.get("_dump_function_fragment") is True
        signatures = module.metadata.get("function_signatures", {})
        valid_fragment = (
            fragment
            and len(graph_names) == 1
            and hasattr(signatures, "keys")
            and execution_names == set(signatures.keys())
            and graph_names.issubset(execution_names)
        )
        if execution_names != graph_names and not valid_fragment:
            raise IRVerificationError(
                "ExecutionFunction set must exactly cover graph functions: "
                f"expected={sorted(graph_names)}, actual={sorted(execution_names)}.",
                stage=module.stage,
            )

    buffer_plan = None
    legacy_buffer_plan = False
    raw_plan = module.metadata.get("buffer_plan")
    if raw_plan is not None:
        from triton.flagmega.ir.bufferization import (
            LEGACY_BUFFER_PLAN_SCHEMA,
            BufferPlan,
        )

        legacy_buffer_plan = raw_plan.get("schema") == LEGACY_BUFFER_PLAN_SCHEMA
        buffer_plan = BufferPlan.from_data(raw_plan)

    shared_formals = {}
    kernel_names = {value.name for value in module.kernel_definitions}
    shared_capacity = None
    if buffer_plan is not None:
        shared_space = buffer_plan.memory_space_map.get("shared")
        if shared_space is not None:
            shared_capacity = shared_space.maximum_bytes
    call_ids: set[str] = set()
    for function in module.execution_functions:
        if buffer_plan is not None:
            try:
                function_plan = buffer_plan.function_map[function.name]
            except KeyError as error:
                raise IRVerificationError(
                    f"ExecutionFunction @{function.name} has no BufferPlan ABI.",
                    stage=module.stage,
                ) from error
            expected_parameters = tuple(
                buffer_id
                for _, buffers in function_plan.parameters
                for buffer_id in buffers
            )
            expected_results = tuple(
                buffer_id
                for _, buffers in function_plan.outputs
                for buffer_id in buffers
            )
            if function.parameters != expected_parameters or function.results != expected_results:
                raise IRVerificationError(
                    f"ExecutionFunction @{function.name} signature differs from "
                    "its BufferPlan ABI.",
                    stage=module.stage,
                )
        function_calls = execution_calls_of(function)
        for call in function_calls:
            if isinstance(call, KernelInvoke) and call.kernel not in kernel_names:
                raise IRVerificationError(
                    f"KernelInvoke {call.call_id!r} references missing kernel definition @{call.kernel}.",
                    stage=module.stage, node_id=call.call_id,
                )
            if isinstance(call, PrimFunctionCall) and call.callee in kernel_names:
                raise IRVerificationError(
                    f"PrimFunctionCall {call.call_id!r} cannot call a non-function kernel definition; use KernelInvoke.",
                    stage=module.stage, node_id=call.call_id,
                )
            if call.call_id in call_ids:
                raise IRVerificationError(
                    f"Duplicate execution call id {call.call_id!r}.",
                    stage=module.stage,
                    node_id=call.call_id,
                )
            call_ids.add(call.call_id)
            primitive = module.kernel_callable_map.get(call.callee)
            callee = functions.get(call.callee)
            if primitive is None and callee is None:
                raise IRVerificationError(
                    f"Execution call {call.call_id!r} references missing callee "
                    f"@{call.callee}.",
                    stage=module.stage,
                    node_id=call.call_id,
                )
            if primitive is not None and call.memory_pools:
                raise IRVerificationError(
                    f"Kernel execution call {call.call_id!r} cannot own a nested "
                    "function workspace frame.",
                    stage=module.stage,
                    node_id=call.call_id,
                )
            if primitive is not None:
                expected_arguments = tuple(
                    buffer.name
                    for parameter in primitive.runtime_parameters
                    for buffer in parameter.buffers
                )
                expected_results = tuple(
                    buffer.name
                    for parameter in primitive.output_parameters
                    for buffer in parameter.buffers
                )
                expected_workspaces = tuple(
                    buffer.name
                    for parameter in primitive.workspaces
                    for buffer in parameter.buffers
                )
            else:
                expected_arguments = callee.parameters
                expected_results = callee.results
                # Nested call frames are represented by caller-owned BufferPlan
                # storage, not logical result/argument edges.
                expected_workspaces = tuple(value.formal for value in call.workspaces)
            actual_arguments = tuple(value.formal for value in call.arguments)
            actual_results = tuple(value.formal for value in call.results)
            actual_workspaces = tuple(value.formal for value in call.workspaces)
            if actual_arguments != expected_arguments:
                raise IRVerificationError(
                    f"Execution call {call.call_id!r} argument ABI differs from "
                    f"@{call.callee}: expected={expected_arguments}, "
                    f"actual={actual_arguments}.",
                    stage=module.stage,
                    node_id=call.call_id,
                )
            if actual_results != expected_results:
                raise IRVerificationError(
                    f"Execution call {call.call_id!r} result ABI differs from "
                    f"@{call.callee}: expected={expected_results}, "
                    f"actual={actual_results}.",
                    stage=module.stage,
                    node_id=call.call_id,
                )
            if actual_workspaces != expected_workspaces:
                raise IRVerificationError(
                    f"Execution call {call.call_id!r} workspace ABI differs from "
                    f"@{call.callee}.",
                    stage=module.stage,
                    node_id=call.call_id,
                )
            for shared in call.shared_workspace_buffers:
                physical = shared.mem_span.buffer
                if (
                    physical.memory_space != "shared"
                    or physical.function != function.name
                ):
                    raise IRVerificationError(
                        f"Execution call {call.call_id!r} shared buffer "
                        f"{shared.name!r} is not owned by caller @{function.name}.",
                        stage=module.stage,
                        node_id=call.call_id,
                    )
            if call.callee not in shared_formals:
                if primitive is not None:
                    dispatch = kernel_dispatch_of(primitive)
                    shared_formals[call.callee] = (
                        () if dispatch is None else dispatch.shared_workspace_buffers
                    )
                else:
                    # Same ordered resource interface as the reusable callee:
                    # consumer traversal sees each semantic call exactly once.
                    shared_formals[call.callee] = tuple({
                        buffer.name: buffer
                        for nested_call in execution_calls_of(callee)
                        for buffer in nested_call.shared_workspace_buffers
                    }.values())
            verify_shared_call_abi(
                call, shared_formals[call.callee], stage=module.stage,
                leaf=primitive is not None, maximum_bytes=shared_capacity,
            )
            if buffer_plan is not None:
                missing = {
                    value.actual
                    for value in (*call.arguments, *call.results, *call.workspaces)
                    if value.actual not in buffer_plan.buffer_map
                }
                if missing:
                    raise IRVerificationError(
                        f"Execution call {call.call_id!r} references BufferPlan "
                        f"objects that do not exist: {sorted(missing)}.",
                        stage=module.stage,
                        node_id=call.call_id,
                    )
                nested_record = buffer_plan.call_map.get(call.call_id)
                if nested_record is not None:
                    expected_pools = {
                        value.memory_space: (
                            value.allocation,
                            value.offset,
                            value.scope_bytes,
                        )
                        for value in nested_record.memory_pools
                    }
                    actual_pools = {
                        value.memory_space: (
                            value.allocation,
                            value.offset,
                            value.nbytes,
                        )
                        for value in call.memory_pools
                    }
                    if legacy_buffer_plan:
                        # Execution TIR v1 stored only the historical workspace
                        # offset/size pair, so its decoded frame has no allocation
                        # identity.  The accompanying v5 BufferPlan still owns
                        # that identity.  Upgrade only this known omission for
                        # comparison; v6 dumps must carry the exact allocation.
                        actual_pools = {
                            memory_space: (
                                (
                                    expected_pools[memory_space][0]
                                    if allocation is None
                                    and memory_space in expected_pools
                                    else allocation
                                ),
                                offset,
                                nbytes,
                            )
                            for memory_space, (
                                allocation,
                                offset,
                                nbytes,
                            ) in actual_pools.items()
                        }
                    if actual_pools != expected_pools:
                        raise IRVerificationError(
                            f"Execution call {call.call_id!r} memory-pool frames "
                            "differ from BufferPlan.",
                            stage=module.stage,
                            node_id=call.call_id,
                        )

        _verify_barrier_topology(function)
        _verify_call_topology(function, function_calls)

    if buffer_plan is not None and module.execution_functions:
        expected_calls = set(buffer_plan.call_map) | set(buffer_plan.kernel_call_map)
        if call_ids != expected_calls:
            raise IRVerificationError(
                "ExecutionFunction call closure differs from BufferPlan: "
                f"missing={sorted(expected_calls - call_ids)}, "
                f"extra={sorted(call_ids - expected_calls)}.",
                stage=module.stage,
            )


def _verify_barrier_topology(function: ExecutionFunction) -> None:
    seen: set[str] = set()
    pending: list[Barrier] = []
    for item in execution_items_of(function):
        if isinstance(item, Barrier):
            if not item.after or not set(item.after).issubset(seen):
                raise IRVerificationError(
                    f"Barrier before {item.before!r} in @{function.name} "
                    "references a call that has not executed."
                )
            pending.append(item)
            continue
        if not isinstance(item, (PrimFunctionCall, KernelInvoke)):
            continue
        if pending and any(value.before != item.call_id for value in pending):
            raise IRVerificationError(
                f"Barrier in @{function.name} is not immediately attached to "
                f"execution call {item.call_id!r}."
            )
        pending.clear()
        seen.add(item.call_id)
    if pending:
        raise IRVerificationError(
            f"ExecutionFunction @{function.name} ends with an unattached barrier."
        )


def _verify_call_topology(
    function: ExecutionFunction,
    calls: tuple[PrimFunctionCall, ...],
) -> None:
    """Verify first-class SSA/effect predecessors in execution order."""

    call_ids = {call.call_id for call in calls}
    seen: set[str] = set()
    for call in calls:
        unknown = set(call.dependencies) - call_ids
        if unknown:
            raise IRVerificationError(
                f"Execution call {call.call_id!r} in @{function.name} has "
                f"unknown dependencies {sorted(unknown)}."
            )
        missing = set(call.dependencies) - seen
        if missing:
            raise IRVerificationError(
                f"Execution call {call.call_id!r} in @{function.name} executes "
                f"before dependencies {sorted(missing)}."
            )
        seen.add(call.call_id)


__all__ = ["execution_calls_of", "execution_items_of", "verify_execution_functions"]
